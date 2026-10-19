"""Fetch recorded upstream revisions, apply recorded patches and verify the kit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from pipeline import read,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--dest',type=Path,required=True);p.add_argument('--kit-archive',type=Path)
    a=p.parse_args();a.dest.mkdir(parents=True,exist_ok=True)
    for dep in read(REPO/'dependencies.json')['repositories']:
        target=a.dest/dep['name']
        if target.exists():raise ValueError('Use an empty destination; will not alter '+str(target))
        patch=REPO/dep['patch']
        if digest(patch)!=dep['patch_sha256']:raise ValueError('Patch changed')
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',dep['remote'],str(target)],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',dep['commit']],check=True)
        subprocess.run(['git','-C',str(target),'submodule','update','--init','--recursive'],check=True)
        if patch.stat().st_size:
            subprocess.run(['git','-C',str(target),'apply','--check',str(patch)],check=True)
            subprocess.run(['git','-C',str(target),'apply',str(patch)],check=True)
    kit=read(REPO/'provenance/official_kit.json');archive=a.kit_archive or a.dest/'official_submission_kit.zip'
    if not archive.exists():urllib.request.urlretrieve(kit['url'],archive)
    if digest(archive)!=kit['archive_sha256']:
        raise ValueError('The live kit URL changed. Supply the recorded archive using --kit-archive; do not silently change packers.')
    target=a.dest/'v2d_submission_kit_20261018';target.mkdir()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():continue
            rel=Path(info.filename)
            if str(rel) not in kit['files'] and len(rel.parts)>1:rel=Path(*rel.parts[1:])
            if str(rel) not in kit['files'] or rel.is_absolute() or '..' in rel.parts:raise ValueError('Unexpected kit member')
            data=z.read(info)
            if hashlib.sha256(data).hexdigest()!=kit['files'][str(rel)]:raise ValueError('Kit member checksum mismatch')
            f=target/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
    print(json.dumps({'dependencies':str(a.dest.absolute()),'official_kit':str(target.absolute())}))


if __name__=='__main__':main()
