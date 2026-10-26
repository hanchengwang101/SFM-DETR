"""Validate a COCO detection directory before training."""
import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('root'); a=ap.parse_args(); r=Path(a.root); ok=True
 for split in ('train','val'):
  ann=r/'annotations'/f'instances_{split}2017.json'; imgs=r/f'{split}2017'
  d=json.loads(ann.read_text(encoding='utf-8')); missing=[x['file_name'] for x in d['images'] if not (imgs/x['file_name']).is_file()]
  print(f'{split}: images={len(d["images"])} annotations={len(d["annotations"])} categories={len(d["categories"])} missing={len(missing)}')
  ok &= not missing
 if not ok: raise SystemExit(1)
if __name__=='__main__': main()
