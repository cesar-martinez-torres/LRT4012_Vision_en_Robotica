"""Package the audited dataset for class without repeating thousands of downloads."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from prepare_dataset import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/ppe_four')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/ppe_four.zip')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads((args.dataset/'audit_report.json').read_text())
    manifest_hash=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest()
    if report['status']!='PASS' or report['manifest_sha256']!=manifest_hash:
        raise ValueError('Audit the dataset before packaging.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(args.dataset.rglob('*')):
            if path.is_file() and path.suffix not in {'.cache','.pyc'}:archive.write(path,path.relative_to(args.dataset).as_posix())
        archive.write(ROOT/'FUENTES.md','FUENTES.md')
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.sha256').write_text(f'{digest}  {args.output.name}\n',encoding='utf-8')
    print(json.dumps({'archive':str(args.output),'bytes':args.output.stat().st_size,'sha256':digest},indent=2))


if __name__=='__main__':main()
