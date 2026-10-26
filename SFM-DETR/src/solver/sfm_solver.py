"""SFM experiment protocol, using the original RT-DETR training/evaluation engine."""
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .det_solver import DetSolver
from .det_engine import train_one_epoch, evaluate
from .sfm_logging import (now, atomic_json, atomic_text, read_records, commit_record,
                          evaluation_data, validate_records)
from ..misc import dist_utils


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SFMDetSolver(DetSolver):
    def train(self):
        super().train()
        self.train_dataloader.worker_init_fn = seed_worker
        self.val_dataloader.worker_init_fn = seed_worker
        self.best_stat = getattr(self, 'best_stat', {'epoch': -1, 'coco_eval_bbox': -1.})
        self.run_id = self.cfg.yaml_cfg['sfm_run_id']
        self.variant = self.cfg.yaml_cfg['sfm_variant']
        optimized = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        assert optimized == {id(p) for p in self.model.parameters() if p.requires_grad}
        self.output_dir.joinpath('checkpoints').mkdir(exist_ok=True)

    def load_tuning_state(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        source = state['ema']['module'] if 'ema' in state else state['model']
        target = self.model.state_dict()
        matched, info = self._matched_state(target, source)
        self.model.load_state_dict(matched, strict=False)
        allowed = ('decoder.denoising_class_embed', 'decoder.enc_score_head', 'decoder.dec_score_head',
                   'decoder.anchors', 'decoder.valid_mask')
        bad = [k for k in info['missed'] + info['unmatched'] if not k.startswith(('encoder.scwa.', 'encoder.mfa.') + allowed)]
        if bad:
            raise RuntimeError(f'Unexpected pretrained loading omissions: {bad}')
        common = {k: v for k, v in self.model.state_dict().items() if not k.startswith(('encoder.scwa.', 'encoder.mfa.'))}
        atomic_json(Path(self.cfg.output_dir) / 'weight_loading.json', dict(
            loaded_keys=sorted(matched), missing_keys=info['missed'], shape_mismatches={k: dict(
                source=list(source[k].shape), target=list(target[k].shape)) for k in info['unmatched']},
            unexpected_keys=sorted(set(source)-set(target)), common_initial_state_sha256=tensor_hash(common),
            regenerated_buffers='decoder.anchors/valid_mask depend on 1024 input, not learned weights',
            source=str(path), passed=True))
        print(f'Pretrained audit passed: {len(matched)} loaded; {len(info["missed"])} new; {len(info["unmatched"])} class-dependent.')

    def state_dict(self):
        state = super().state_dict()
        state['sfm_best_stat'] = self.best_stat
        state['sfm_pending'] = getattr(self, 'pending', None)
        state['sfm_rng'] = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        state['sfm_config_hash'] = self.cfg.yaml_cfg['sfm_config_hash']
        state['sfm_loader'] = dict(epoch=self.last_epoch, persistent_workers=False,
                                    generator='global torch RNG; random sampler and worker base seeds restored at epoch boundary')
        return state

    def load_state_dict(self, state):
        if state.get('sfm_config_hash') != self.cfg.yaml_cfg['sfm_config_hash']:
            raise RuntimeError('Refusing resume with a different SFM config hash')
        super().load_state_dict(state)
        self.best_stat = state['sfm_best_stat']
        self.pending = state.get('sfm_pending')
        rng = state['sfm_rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if rng['cuda'] is not None:
            torch.cuda.set_rng_state_all(rng['cuda'])

    def save_checkpoint(self, path):
        if dist_utils.is_main_process():
            path = Path(path)
            temp = path.with_name(path.name+'.tmp')
            torch.save(self.state_dict(), temp)
            os.replace(temp, path)

    def _eval(self):
        module = self.ema.module if self.ema else self.model
        # All variants use the original fp32 EMA evaluation path.
        return evaluate(module, self.criterion, self.postprocessor, self.val_dataloader, self.evaluator, self.device)

    def fit(self):
        self.train()
        expected = self.cfg.epoches
        records = read_records(self.output_dir)
        if records and records[-1]['epoch_zero_based'] > self.last_epoch:
            raise RuntimeError('Metrics lead checkpoint; preserve this directory and investigate instead of overwriting.')
        logged = records[-1]['epoch_zero_based'] if records else -1
        if self.last_epoch > logged:
            if self.last_epoch != logged+1 or not self.pending:
                raise RuntimeError('Unrecoverable log gap; do not fabricate historical losses')
            # Checkpoint may have been committed immediately before log write.
            rng = self.state_dict()['sfm_rng']
            stats, ev = self._eval()
            recovered = self.pending.copy()
            recovered.update(evaluation_data(stats, ev, self.run_id, self.last_epoch+1))
            recovered['recovered_by_checkpoint_validation'] = True
            commit_record(self.output_dir, recovered)
            random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
            if rng['cuda'] is not None:
                torch.cuda.set_rng_state_all(rng['cuda'])
        for epoch in range(self.last_epoch+1, expected):
            self.train_dataloader.set_epoch(epoch)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            lr_start = [g['lr'] for g in self.optimizer.param_groups]
            train = train_one_epoch(self.model, self.criterion, self.train_dataloader, self.optimizer,
                                    self.device, epoch, max_norm=self.cfg.clip_max_norm,
                                    print_freq=self.cfg.print_freq, ema=self.ema, scaler=self.scaler,
                                    lr_warmup_scheduler=self.lr_warmup_scheduler, writer=self.writer)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            train_end = time.perf_counter()
            lr_end = [g['lr'] for g in self.optimizer.param_groups]
            # Advance once per epoch, including warmup epochs: drop after epoch 90.
            self.lr_scheduler.step()
            self.last_epoch = epoch
            stats, evaluator = self._eval()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            ended = time.perf_counter()
            data = evaluation_data(stats, evaluator, self.run_id, epoch+1)
            is_best = data['map50_95'] > self.best_stat['coco_eval_bbox']
            if is_best:
                self.best_stat = dict(epoch=epoch, coco_eval_bbox=data['map50_95'])
            raw = {**{'train_'+k: v for k, v in train.items()}, **{'test_'+k: v for k, v in stats.items()}, 'epoch': epoch}
            record = dict(run_id=self.run_id, variant=self.variant, seed=self.cfg.seed, epoch=epoch+1,
                          epoch_zero_based=epoch, split='val', eval_weight='ema', learning_rate=lr_end[0],
                          lr_groups_json=json.dumps(lr_end), lr_groups_start=lr_start,
                          lr_groups_next=[g['lr'] for g in self.optimizer.param_groups],
                          train_loss=train['loss'], train_loss_vfl=train.get('loss_vfl'),
                          train_loss_bbox=train.get('loss_bbox'), train_loss_giou=train.get('loss_giou'),
                          train_loss_components_weighted=True, raw_log=raw,
                          epoch_train_seconds=train_end-started, epoch_val_seconds=ended-train_end,
                          epoch_total_seconds=ended-started,
                          gpu_peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2 if self.device.type == 'cuda' else 0.,
                          best_map50_95=self.best_stat['coco_eval_bbox'], best_epoch=self.best_stat['epoch']+1,
                          is_best=is_best, completed_at=now(), **data)
            self.pending = record
            self.save_checkpoint(self.output_dir/'last.pth')
            if is_best:
                self.save_checkpoint(self.output_dir/'best.pth')
            if (epoch+1) % self.cfg.checkpoint_freq == 0:
                self.save_checkpoint(self.output_dir/'checkpoints'/f'epoch_{epoch+1:03d}.pth')
            if dist_utils.is_main_process():
                commit_record(self.output_dir, record)
                atomic_json(self.output_dir/'status.json', dict(state='running', epoch=epoch+1, total_epochs=expected,
                            best_epoch=self.best_stat['epoch']+1, best_map50_95=self.best_stat['coco_eval_bbox'], updated_at=now()))
            print(f'SFM epoch {epoch+1}/{expected}, AP={data["map50_95"]:.6f}, best={self.best_stat}, seconds={ended-started:.1f}', flush=True)
        validate_records(self.output_dir, expected)
        atomic_json(self.output_dir/'status.json', dict(state='completed', epoch=expected, total_epochs=expected,
                    best_epoch=self.best_stat['epoch']+1, best_map50_95=self.best_stat['coco_eval_bbox'], updated_at=now()))

    def val(self):
        self.eval()
        stats, evaluator = self._eval()
        data = evaluation_data(stats, evaluator, self.cfg.yaml_cfg['sfm_run_id'], self.last_epoch+1)
        atomic_json(self.output_dir/self.cfg.yaml_cfg.get('sfm_eval_filename', 'validation_metrics.json'), dict(
                    **data, epoch=self.last_epoch+1, eval_weight='ema', raw_stats=stats,
                    split=self.cfg.yaml_cfg.get('sfm_eval_split', 'val')))
