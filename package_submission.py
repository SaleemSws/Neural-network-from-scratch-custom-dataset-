"""Create portable local submission ZIP including data and measured artifacts."""
import argparse
import zipfile
from pathlib import Path
from common import ROOT, save_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT.parent/'Assignment-10-lane-seg-submission-65-35.zip')
    args=p.parse_args()
    count=0
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or any(x in path.parts for x in ('.git','__pycache__','.venv')) or path.suffix=='.zip':
                continue
            if path.resolve()==args.output.resolve():
                continue
            archive.write(path,Path(ROOT.name)/path.relative_to(ROOT));count+=1
    with zipfile.ZipFile(args.output) as archive:
        bad=archive.testzip()
        if bad:raise RuntimeError(f'ZIP integrity failed: {bad}')
    print(f'{args.output}: {count} files, {args.output.stat().st_size/1024**2:.2f} MiB; CRC integrity passed')


if __name__=='__main__':
    main()
