"""Download recorded HF/URL model files and verify SHA-256; never print credentials."""
import argparse
from pathlib import Path
import shutil
import sys
import urllib.request

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from pipeline import read,digest,link


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True)
    p.add_argument('--sam3d-objects',type=Path,required=True);p.add_argument('--sam2-checkpoint',type=Path,required=True)
    p.add_argument('--only',default='cari4d_commercial,sam-3d-body-dinov3,sam3d_objects,moge2,sam2')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    from huggingface_hub import hf_hub_download
    selected=set(a.only.split(','))
    for model in read(REPO/'models.json'):
        if model['name'] not in selected:continue
        if model['provider'] in ['upstream-downloader','torch-hub']:
            base=a.models/'cari4d_runtime'/('foundationpose/nvlabs_pytorch' if model['name']=='foundationpose' else 'sam3d_body/torch_home/hub')
        elif model['name']=='sam3d_objects':base=a.sam3d_objects/'checkpoints/hf'
        elif model['name']=='moge2':base=a.models/'cari4d_runtime/hf_home/hub'
        elif model['name']=='sam2':base=a.sam2_checkpoint.parent
        else:base=a.models/model['name']
        for filename,record in model['files'].items():
            if model['name']=='moge2':
                target=base/('models--'+model['repo'].replace('/','--'))/'snapshots'/model['revision']/filename
            elif model['name']=='sam2':target=a.sam2_checkpoint
            else:target=base/record.get('local_name',filename)
            if not target.is_file():
                if a.verify_only:raise FileNotFoundError(target)
                target.parent.mkdir(parents=True,exist_ok=True)
                if model['provider']=='huggingface':
                    cached=hf_hub_download(model['repo'],filename,revision=model['revision'])
                    shutil.copyfile(cached,target)
                elif model['provider']=='huggingface-cache':
                    cached=hf_hub_download(model['repo'],filename,revision=model['revision'],cache_dir=base)
                    if Path(cached).absolute()!=target.absolute():shutil.copyfile(cached,target)
                elif model['provider']=='url':urllib.request.urlretrieve(model['url'],target)
                else:raise FileNotFoundError('Use the upstream downloader described in ENVIRONMENT.md: '+str(target))
            # Local YAML overlays are recorded explicitly, separately from weights.
            overlay=REPO/'model_configs/sam3d_objects'/Path(filename).name
            if model['name']=='sam3d_objects' and target.suffix=='.yaml' and not a.verify_only and overlay.exists():
                shutil.copyfile(overlay,target)
            if target.stat().st_size!=record['size'] or digest(target)!=record['sha256']:
                raise ValueError('Model checksum mismatch: '+str(target))
            print('VERIFIED',model['name'],filename,flush=True)
        if model['name']=='moge2' and not a.verify_only:
            ref=base/('models--'+model['repo'].replace('/','--'))/'refs/main'
            ref.parent.mkdir(parents=True,exist_ok=True);ref.write_text(model['revision'])
    if not a.verify_only:
        # Match the relative asset layout used by the frozen CARI4D driver.
        for source,target in [(a.models/'cari4d_commercial',a.models/'cari4d_runtime/cari4d'),
                              (a.models/'sam-3d-body-dinov3',a.models/'cari4d_runtime/sam3d_body/checkpoints/sam-3d-body-dinov3'),
                              (a.models/'sam-3d-body-dinov3/assets/mhr_model.pt',a.models/'mhr_model.pt')]:
            if source.exists() and not target.exists():link(source,target)


if __name__=='__main__':main()
