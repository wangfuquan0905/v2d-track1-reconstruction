"""Validate official rows and roundtrip every packed parameter and reviewed mesh."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/track1_safe_windows_v1'
KIT=ROOT/'third_party/v2d_submission_kit_20261018';sys.path.insert(0,str(KIT))
from submit.v2d_submit import validate_submission
from v2dlb import mhr_submission as ms


def main():
    p=argparse.ArgumentParser();p.add_argument('parquet',type=Path);a=p.parse_args()
    frame=validate_submission(a.parquet,'track_1',json.loads((KIT/'config/tracks.json').read_text()))
    parts=frame['row_id'].str.split('_',expand=True)
    keys=parts[[1,2,3,4]].astype(int);keys.columns=['episode','frame','array','point']
    values=frame[['x','y','z']].to_numpy(dtype=np.float64);rows=[];total=0
    protocol=json.loads((ROOT/'configs/track1_test_frames_v1.json').read_text())
    scored={r['episode_index']:np.array(r['scored_frames']) for r in protocol['episodes']}
    for episode,group in keys.groupby('episode'):
        name=f'episode_{episode:06d}';ids=np.sort(group.loc[group['frame']!=ms.EPISODE_FRAME,'frame'].unique())
        np.testing.assert_array_equal(ids,scored[episode]);total+=len(ids)
        with np.load(OUT/'submission_episodes'/(name+'.npz')) as raw,np.load(OUT/'submission_meshes'/(name+'.npz')) as mesh:
            expected={'mhr_pose':raw['pose'][ids],'mhr_scales':raw['scales'],'mhr_shape':raw['shape'],
                'object_rotation':raw['object_rotation'][ids],'object_translation':raw['object_translation'][ids],
                'object_scale':np.array([[float(raw['object_scale']),0,0]]),
                'object_mesh_vertices':mesh['vertices'],'object_mesh_faces':mesh['faces']}
            for index,key in enumerate(ms.array_order()):
                selection=group[group['array']==index].sort_values(['frame','point'])
                block=values[selection.index]
                if key in ms.PARAMETER_COUNTS:
                    count=ms.PARAMETER_COUNTS[key];n=len(ids) if key=='mhr_pose' else 1
                    block=block.reshape(n,-1)
                    np.testing.assert_array_equal(block[:,count:],0)
                    block=block[:,:count].reshape(expected[key].shape)
                else:block=block.reshape(expected[key].shape)
                np.testing.assert_array_equal(block,expected[key],err_msg=f'{name} {key} roundtrip mismatch')
        rows.append({'episode':name,'scored_frames':len(ids),'parameters_and_reviewed_mesh':'exact_match'})
    if len(rows)!=30 or total!=9877:raise ValueError('Incomplete competition coverage')
    result={'status':'passed','parquet':str(a.parquet.resolve()),'sha256':hashlib.sha256(a.parquet.read_bytes()).hexdigest(),
        'rows':len(frame),'episodes':rows,'scored_frames':total,'code_commit_url':str(frame['code_commit_url'].iloc[0]),
        'row_order':'official sample exact match','all_parameters':'staged arrays exact match',
        'mesh_geometry':'3D review mesh exact match','uploaded':False}
    (OUT/'packed_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':main()
