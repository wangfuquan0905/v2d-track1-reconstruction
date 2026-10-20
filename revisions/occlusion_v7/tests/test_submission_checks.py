import numpy as np
import pytest
from v2d_syilen.submission_checks import check_episode, submission_arrays


def episode():
    return dict(frames=np.arange(4),pose=np.zeros((4,136)),scales=np.zeros(68),shape=np.zeros(45),
        rotation=np.tile(np.eye(3),(4,1,1)),translation=np.zeros((4,3)),object_scale=np.array(1.),
        vertices=np.zeros((4,3,3)),joints=np.zeros((4,2,3)))


def test_sparse_scoring_preserves_original_video_indices():
    data=episode();data['translation'][:,0]=np.arange(4)
    check_episode(data,4,[1,3],baseline=data)
    packed=submission_arrays(data)
    np.testing.assert_array_equal(packed['object_translation'][:,0],[0,1,2,3])


def test_invalid_submission_geometry_and_frame_coverage_rejected():
    for mutation in [lambda d:d.update(frames=np.array([0,1,1,3])),
                     lambda d:d['rotation'].__setitem__((1,0,0),-1.),
                     lambda d:d['translation'].__setitem__((1,0),np.nan),
                     lambda d:d.update(object_scale=np.array(0.))]:
        data=episode();mutation(data)
        with pytest.raises((ValueError,AssertionError)):check_episode(data,4,[1,3])
    with pytest.raises(ValueError):check_episode(episode(),4,[1,4])
    data=episode();base=episode();data['pose'][2,0]=.1
    with pytest.raises(AssertionError):check_episode(data,4,[1,3],baseline=base)
