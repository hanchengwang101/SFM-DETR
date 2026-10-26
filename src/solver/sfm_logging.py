"""Atomic experiment records. JSONL is the source of truth for derived CSVs."""
import csv
import io
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np

FIELDS = ('run_id variant seed epoch epoch_zero_based split eval_weight learning_rate lr_groups_json '
          'train_loss train_loss_vfl train_loss_bbox train_loss_giou precision recall f1 map50 map50_95 map75 '
          'map_small map_medium map_large ar1 ar10 ar100 epoch_train_seconds epoch_val_seconds '
          'epoch_total_seconds gpu_peak_memory_mb best_map50_95 best_epoch is_best completed_at').split()
PR_DEFINITION = 'coco_macro_interpolated_pr_max_f1_iou50'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def write_csv(path, rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def read_records(run):
    path = Path(run) / 'metrics.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def commit_record(run, record):
    run = Path(run)
    records = {(r['run_id'], r['epoch']): r for r in read_records(run)}
    records[(record['run_id'], record['epoch'])] = record
    records = sorted(records.values(), key=lambda r: r['epoch'])
    atomic_text(run / 'metrics.jsonl', ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records))
    rebuild_csv(run, records)


def rebuild_csv(run, records=None):
    run = Path(run)
    records = read_records(run) if records is None else records
    write_csv(run / 'metrics.csv', records, FIELDS)
    per_class = [row for r in records for row in r['per_class']]
    write_csv(run / 'per_class_metrics.csv', per_class,
              ['run_id', 'epoch', 'category_id', 'category_name', 'map50_95', 'map50', 'map75', 'box_count'])
    atomic_text(run / 'log.txt', ''.join(json.dumps(r['raw_log'], allow_nan=False)+'\n' for r in records))


def finite_mean(values):
    values = np.asarray(values)
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else None


def evaluation_data(stats, evaluator, run_id, epoch):
    ev = evaluator.coco_eval['bbox']
    ap = stats['coco_eval_bbox']
    names = ['map50_95', 'map50', 'map75', 'map_small', 'map_medium', 'map_large', 'ar1', 'ar10', 'ar100']
    data = {name: float(ap[i]) if ap[i] >= 0 else None for i, name in enumerate(names)}
    p, r = stats['precision'], stats['recall']
    data.update(precision=p, recall=r, f1=2*p*r/max(p+r, 1e-12) if p is not None and r is not None else None)
    tensor = ev.eval['precision']
    per_class = []
    for i, cat_id in enumerate(ev.params.catIds):
        cat = ev.cocoGt.cats[cat_id]
        per_class.append(dict(run_id=run_id, epoch=epoch, category_id=int(cat_id), category_name=cat['name'],
                              map50_95=finite_mean(tensor[:, :, i, 0, -1]),
                              map50=finite_mean(tensor[np.isclose(ev.params.iouThrs, .5), :, i, 0, -1]),
                              map75=finite_mean(tensor[np.isclose(ev.params.iouThrs, .75), :, i, 0, -1]),
                              box_count=len(ev.cocoGt.getAnnIds(catIds=[cat_id]))))
    data['per_class'] = per_class
    data['evaluation_protocol'] = dict(precision_recall=PR_DEFINITION, max_dets=list(ev.params.maxDets),
                                       iou_grid=ev.params.iouThrs.tolist(), recall_grid=ev.params.recThrs.tolist(),
                                       undefined_metrics={n: 'COCO has no eligible ground truth' for n in names if data[n] is None})
    return data


def validate_records(run, expected=100):
    records = read_records(run)
    errors = []
    if [r['epoch'] for r in records] != list(range(1, expected+1)):
        errors.append(f'Expected continuous 1..{expected}; got {[r["epoch"] for r in records]}')
    for r in records:
        for key in ['precision', 'recall', 'f1', 'map50', 'map50_95', 'map75', 'train_loss']:
            if r.get(key) is None or not np.isfinite(r[key]):
                errors.append(f'epoch {r["epoch"]}: invalid {key}')
    if errors:
        raise RuntimeError('; '.join(errors))
    return records
