"""Create a portable SFM-DETR YAML from a release template."""
from pathlib import Path
import argparse, os, re

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--output-dir', default='./runs/sfm_detr')
    ap.add_argument('--device', default='cuda:0')
    args=ap.parse_args()
    text=Path(args.template).read_text(encoding='utf-8')
    data_root = args.data_root.replace('\\', '/')
    text=text.replace('__DATA_ROOT_WTBD__', data_root).replace('__DATA_ROOT_WTBS2025__', data_root)
    text=re.sub(r'^output_dir:.*$', f'output_dir: {args.output_dir}', text, flags=re.M)
    text=re.sub(r'^device:.*$', f'device: {args.device}', text, flags=re.M)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(text,encoding='utf-8')
if __name__=='__main__': main()
