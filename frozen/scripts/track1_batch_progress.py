"""Print a compact read-only snapshot of the Track 1 batch."""
from collections import Counter
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/track1_safe_windows_v1'
state=json.loads((OUT/'status.json').read_text())
counts=Counter(row['status'] for row in state['episodes'].values())
result={'phase':state['phase'],'queue_status':state['status'],'counts':dict(counts),'current':[],'failed':[]}
for ep,row in state['episodes'].items():
    if row['status']=='failed':result['failed'].append({'episode':ep,'stage':row.get('stage'),'error':row.get('error')})
    if row['status']!='running':continue
    stage=row['stage'];meta=OUT/'stages'/ep/(stage+'.json');path=meta.with_suffix('.log')
    native=ROOT/'outputs'/ep/'cari4d'/ep/f'native_{stage}.log'
    if native.exists():path=native
    info=json.loads(meta.read_text())
    with open(path,'rb') as f:
        f.seek(max(0,path.stat().st_size-3000));lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
    result['current'].append({'episode':ep,'stage':stage,'elapsed_seconds':round(time.time()-info['started']),
        'last_output':lines[-1][-350:] if lines else ''})
print(json.dumps(result,indent=2))
