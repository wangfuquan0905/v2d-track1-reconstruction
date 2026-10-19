"""Conservative launch checks after the public-data validation caused host OOM.

This is admission control, not a runtime memory limit or a guarantee of fit.
Uses only the standard library and nvidia-smi; does not initialize CUDA.
"""
from pathlib import Path
import os
import subprocess


def memory_issues(info, pressure, bounded):
    issues=[]
    minimum=18 if bounded else 20
    if info['MemAvailable'] < minimum:
        issues.append(f"available RAM {info['MemAvailable']:.1f} GiB < {minimum} GiB minimum")
    if pressure.get('some',0)>5 or pressure.get('full',0)>1:
        issues.append(f'active memory pressure: {pressure}')
    swap_used=info['SwapTotal']-info['SwapFree']
    # Cold pages can remain swapped out long after pressure has subsided.
    # High occupancy alone is acceptable only inside the enforced small cgroup.
    if info['SwapTotal'] and swap_used/info['SwapTotal']>.5 and not bounded:
        issues.append('high swap occupancy requires MemoryMax<=18 GiB and MemorySwapMax<=512 MiB')
    return issues


def memory_context():
    pressure={}
    try:
        for line in Path('/proc/pressure/memory').read_text().splitlines():
            fields=line.split();pressure[fields[0]]=float(fields[1].split('=')[1])
    except (OSError,ValueError):
        return {},False
    try:
        group=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
        path=Path('/sys/fs/cgroup')/group.lstrip('/')
        maximum=int((path/'memory.max').read_text());swap=int((path/'memory.swap.max').read_text())
        bounded=maximum<=18*1024**3 and swap<=512*1024**2
    except (OSError,ValueError,StopIteration):bounded=False
    return pressure,bounded


def require_reconstruction_capacity():
    if os.environ.get('V2D_RESOURCE_POLICY')=='unrestricted':
        return  # Explicit user request: run until an actual allocation failure.
    info = {line.split(':')[0]: int(line.split()[1]) / 1024**2
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.split(':')[0] in ('MemAvailable', 'SwapTotal', 'SwapFree')}
    pressure,bounded=memory_context()
    issues = memory_issues(info,pressure,bounded)
    swap_used = info['SwapTotal'] - info['SwapFree']
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=10)
    rows = [tuple(x.strip() for x in line.split(',')) for line in result.stdout.splitlines()]
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
    matches = [r for r in rows if selected in r[:2]]
    if len(matches) != 1:
        issues.append(f"cannot identify selected GPU {selected!r}; refusing to guess")
    elif float(matches[0][2]) / 1024 < 22:
        issues.append(f"available GPU memory {float(matches[0][2])/1024:.1f} GiB < conservative 22 GiB minimum")
    if issues:
        raise RuntimeError('Heavy reconstruction not started: ' + '; '.join(issues) +
                           '. No automatic retry. These thresholds are conservative admission checks, not measured peak budgets.')
    if info['SwapTotal'] and swap_used/info['SwapTotal']>.5:
        print(f'Resource check passed under cgroup cap: available RAM={info["MemAvailable"]:.1f} GiB, swap used={swap_used:.1f} GiB, memory pressure={pressure}',flush=True)


if __name__ == '__main__':
    require_reconstruction_capacity()
    print('Launch preflight passed; runtime memory containment is still required before retrying the failed workload.')
