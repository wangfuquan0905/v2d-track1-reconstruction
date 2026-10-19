"""Run the frozen safe_windows_v1 with a persistent, checked antialias topology hash."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import refine_object_windows_safe as frozen
import nvdiffrast.torch as dr
from v2d_syilen.render_topology_cache import AntialiasTopologyCache


def main():
    out=Path(sys.argv[sys.argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
    cache=AntialiasTopologyCache(dr.antialias,dr.antialias_construct_topology_hash)
    sources=[Path(__file__),ROOT/'v2d_syilen/render_topology_cache.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    dr.antialias=cache
    succeeded=False
    try:
        frozen.main();succeeded=True
    finally:
        dr.antialias=cache.original
        proof={**cache.report(),'source_sha256':hashes,'completed':succeeded}
        (out/'topology_cache_runtime.json').write_text(json.dumps(proof,indent=2))
    config=json.loads((out/'config.json').read_text());config['source_sha256'].update(hashes)
    config['runtime_topology_cache']=proof
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for p in sources:(out/'source_snapshot'/p.name).write_bytes(p.read_bytes())
    print('TOPOLOGY_CACHE_RUNTIME',json.dumps(proof),flush=True)


if __name__=='__main__':main()
