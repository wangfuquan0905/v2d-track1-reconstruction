"""Strict public Track 1 episode checks before packaging any prediction."""
import numpy as np


def check_episode(data, frames, scored_frames, baseline=None):
    expected=np.arange(frames)
    np.testing.assert_array_equal(data['frames'],expected,err_msg='Missing or misindexed video frames')
    shapes={'pose':(frames,136),'scales':(68,),'shape':(45,),
        'rotation':(frames,3,3),'translation':(frames,3)}
    for key,shape in shapes.items():
        a=np.asarray(data[key])
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'{key}: invalid shape or nonfinite values')
    r=np.asarray(data['rotation']);np.testing.assert_allclose(r@r.transpose(0,2,1),np.broadcast_to(np.eye(3),r.shape),atol=2e-5,rtol=0)
    np.testing.assert_allclose(np.linalg.det(r),1.,atol=2e-5,rtol=0)
    scale=float(np.asarray(data['object_scale']).reshape(()))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid object scale')
    ids=np.asarray(scored_frames)
    if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids))!=len(ids) or not np.all((ids>=0)&(ids<frames)):
        raise ValueError('Invalid scored frame coverage')
    if baseline is not None:
        for key in ['frames','pose','shape','scales','object_scale','vertices','joints']:
            np.testing.assert_array_equal(data[key],baseline[key],err_msg=f'Fixed baseline {key} changed')
    return {'frames':frames,'scored_frames':len(ids),'native_parameter_shapes_valid':True,
        'transforms_finite_and_rigid':True,'fixed_human_and_scale':baseline is not None,
        'rotation_max_orthogonality_error':float(np.max(np.abs(r@r.transpose(0,2,1)-np.eye(3))))}


def submission_arrays(data):
    return {**{k:np.asarray(data[k]) for k in ['pose','scales','shape','object_scale']},
        'object_rotation':np.asarray(data['rotation']),'object_translation':np.asarray(data['translation'])}
