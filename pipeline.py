"""Portable entry points around the unmodified, archived producing code."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parent
ORIGINAL = '/home/lab-2/Research/V2D'


def read(path): return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def verify(source_root=None, results_root=None):
    manifest = read(REPO / 'provenance/source_manifest.json')['files']
    for name, expected in manifest.items():
        if digest(REPO / 'frozen' / name) != expected: raise ValueError('Frozen source changed: ' + name)
        if name.endswith('.py'): ast.parse((REPO / 'frozen' / name).read_text())
        if source_root and digest(source_root / name) != expected:
            raise ValueError('Producing source differs: ' + name)
    for dep in read(REPO / 'dependencies.json')['repositories']:
        if digest(REPO / dep['patch']) != dep['patch_sha256']: raise ValueError('Dependency patch changed')
    legacy_files = 0
    for hashes in (REPO / 'provenance/episode16').glob('*/source_hashes.json'):
        for filename, expected in read(hashes).items():
            target = REPO / 'legacy/episode16' / hashes.parent.name / 'source_snapshot' / Path(filename).name
            if digest(target) != expected: raise ValueError('Historical source changed: ' + str(target))
            legacy_files += 1
    tracker = read(REPO / 'provenance/episode16/pruned_tracking_v4_acc001/config.json')
    for filename, expected in tracker['source_sha256'].items():
        target = REPO / 'legacy/episode16/pruned_tracking_v4_acc001/source_snapshot' / Path(filename).name
        if digest(target) != expected: raise ValueError('Historical tracker source changed')
    count = 0
    if results_root:
        for paths in read(REPO / 'provenance/result_hashes.json').values():
            for filename, expected in paths.items():
                if digest(results_root / filename) != expected: raise ValueError('Result changed: ' + filename)
            count += 1
        for filename, expected in read(REPO / 'provenance/episode16/artifact_hashes.json').items():
            if digest(results_root / filename) != expected: raise ValueError('Historical input changed: ' + filename)
    result = {'status': 'passed', 'frozen_files': len(manifest), 'historical_sources': legacy_files + 2,
              'verified_result_episodes': count}
    print(json.dumps(result)); return result


def replacements(work, runtime):
    return {
        '/home/lab-2/Research/RLSIM/third_party/sam2/checkpoints/sam2.1_hiera_large.pt': str(Path(runtime['sam2_checkpoint']).absolute()),
        '/home/lab-2/Research/RLSIM/sam-3d-objects': str(Path(runtime['sam3d_objects']).absolute()),
        '/home/lab-2/miniconda3/envs/RLSIM': str(Path(runtime['cari4d_python']).absolute().parent.parent),
        '/usr/local/cuda-12.8': str(Path(runtime['cuda_home']).absolute()),
        ORIGINAL: str(work),
    }


def adapt(text, work, runtime):
    for old, new in replacements(work, runtime).items(): text = text.replace(old, new)
    return text


def link(source, target):
    source = Path(source).absolute(); target = Path(target)
    if not source.exists(): raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve(): return
    if target.exists() or target.is_symlink(): raise ValueError('Refusing to replace ' + str(target))
    target.symlink_to(source, target_is_directory=source.is_dir())


def materialize(work, runtime_path):
    verify(); runtime = read(runtime_path)
    if work == REPO or work == Path(ORIGINAL): raise ValueError('Use a separate empty workspace')
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()): raise ValueError('Materialization requires an empty directory')
    for source in (REPO / 'frozen').rglob('*'):
        if not source.is_file() or '__pycache__' in source.parts: continue
        target = work / source.relative_to(REPO / 'frozen'); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(adapt(source.read_text(), work, runtime))
    # The historical queue requires a pre-existing result for episode 16. The
    # portable driver reproduces it separately, then enables the same reuse gate.
    batch = work / 'scripts/run_track1_test_batch.py'
    text = batch.read_text().replace('if i==16:', "if i==16 and os.environ.get('V2D_REUSE_EPISODE16')=='1':")
    batch.write_text(text)
    for label, key in [('cari4d', 'cari4d_python'), ('sam3d-object', 'object_python'), ('submission', 'submission_python')]:
        # Link the complete environment so Python preserves its pyvenv.cfg and site packages.
        link(Path(runtime[key]).absolute().parent.parent, work / ('.venv-' + label))
    for key, target in [('video_to_data', 'third_party/video_to_data'), ('submission_kit', 'third_party/v2d_submission_kit_20261018'),
                        ('models', 'data/models')]: link(runtime[key], work / target)
    out = work / 'outputs/track1_safe_windows_v1'; out.mkdir(parents=True)
    shutil.copy2(REPO / 'provenance/batch/dataset_manifest.json', out / 'dataset_manifest.json')
    # Materialize RGB prompts explicitly; the original download script does not do this.
    for row in read(work / 'configs/track1_test_prompts_v1.json')['episodes']:
        prompts = row['prompts']
        if row['episode_index'] == 16: prompts = read(work / 'configs/episode_000016_prompts.json')
        write(work / 'outputs' / row['episode'] / 'inputs/prompts.json', prompts)
    write(work / 'runtime.local.json', runtime)
    write(work / 'materialization.json', {'source_repository': str(REPO), 'path_replacements': replacements(work, runtime),
        'queue_change': 'Episode 16 reuse is enabled only after its explicit historical reproduction.',
        'source_sha256': {str(p.relative_to(work)): digest(p) for folder in ['scripts','v2d_syilen','configs'] for p in (work/folder).glob('*') if p.is_file()}})
    print('MATERIALIZED', work)


def invoke(work, script, args=(), interpreter='cari4d', env=None):
    python = work / ('.venv-' + interpreter) / 'bin/python'
    command = [str(python), str(work / 'scripts' / script), *map(str, args)]
    subprocess.run(command, cwd=work, env=env, check=True)


def episodes(value):
    result = list(range(30)) if value == 'all' else [int(x) for x in value.split(',')]
    if not result or len(set(result)) != len(result) or any(x not in range(30) for x in result):
        raise ValueError('Expected unique episode indices 0..29 or all')
    return result


def reconstruct(work, selection, dry_run=False):
    selected = episodes(selection); ordinary = [i for i in selected if i != 16]
    if dry_run:
        print(json.dumps({'ordinary_episodes': ordinary, 'legacy_episode16': 16 in selected,
                          'stages': ['segment', 'RGB mask recovery', 'reconstruction', 'scale retry',
                                     'render topology retry', 'legacy16 when selected', 'review/validate']}, indent=2)); return
    out = work / 'outputs/track1_safe_windows_v1'
    if not (out / 'inventory.json').exists(): invoke(work, 'prepare_track1_test_batch.py')
    if ordinary:
        ids = ','.join(map(str, ordinary))
        invoke(work, 'run_track1_test_batch.py', ['--phase', 'segment', '--episodes', ids])
        fixes = read(work / 'configs/track1_mask_recovery_v1.json')['episodes']
        for i in ordinary:
            name = f'episode_{i:06d}'
            if name in fixes: invoke(work, 'recover_track1_object_masks.py', ['--episode', name])
        invoke(work, 'prepare_track1_videos.py')
        # The frozen queue identifies scale and rerender failures from its recorded
        # stage logs; bounded retries activate exactly its existing recovery paths.
        pending = ordinary
        for attempt in range(3):
            try: invoke(work, 'run_track1_test_batch.py', ['--phase', 'full', '--episodes', ','.join(map(str, pending))])
            except subprocess.CalledProcessError: pass
            state = read(out / 'status.json')['episodes']
            pending = [i for i in ordinary if state.get(f'episode_{i:06d}', {}).get('status') != 'complete']
            if not pending: break
        if pending: raise RuntimeError('Episodes remain incomplete: ' + str(pending))
    if 16 in selected:
        subprocess.run([str(work / '.venv-cari4d/bin/python'), str(REPO / 'tools/reproduce_episode16.py'),
                        '--workspace', str(work)], check=True)
    invoke(work, 'prepare_track1_videos.py'); invoke(work, 'prepare_track1_review.py')
    invoke(work, 'prepare_track1_submission.py')


def main():
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='command', required=True)
    v = s.add_parser('verify'); v.add_argument('--source-root', type=Path); v.add_argument('--results-root', type=Path)
    m = s.add_parser('materialize'); m.add_argument('--workspace', type=Path, required=True); m.add_argument('--runtime', type=Path, required=True)
    r = s.add_parser('run'); r.add_argument('--workspace', type=Path, required=True); r.add_argument('--episodes', default='all'); r.add_argument('--dry-run', action='store_true')
    k = s.add_parser('pack'); k.add_argument('--project-root', type=Path, required=True); k.add_argument('--commit-url', required=True)
    a = p.parse_args()
    if a.command == 'verify': verify(a.source_root, a.results_root)
    elif a.command == 'materialize': materialize(a.workspace.absolute(), a.runtime)
    elif a.command == 'run': reconstruct(a.workspace.absolute(), a.episodes, a.dry_run)
    else:
        import re
        if not re.fullmatch(r'https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}', a.commit_url):
            raise ValueError('Expected a real full GitHub commit URL')
        # Original run only: validate every producing byte and every selected result
        # before allowing this repository commit to identify the submitted output.
        verify(a.project_root, a.project_root)
        invoke(a.project_root, 'prepare_track1_submission.py', ['--commit-url', a.commit_url])


if __name__ == '__main__': main()
