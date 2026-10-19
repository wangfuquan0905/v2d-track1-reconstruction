from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'third_party/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d'))
from resource_preflight import memory_issues
from prep.mhr_foundationpose_diagnostics import serialized_centroid_distance,validate_foundationpose_selection_arrays,FOUNDATIONPOSE_SELECTION_DTYPES
from prep.foundationpose_pose_selection import measure_visible_silhouettes,validate_rigid_transform


def test_explicit_unrestricted_policy_skips_resource_probes(monkeypatch):
    import resource_preflight
    monkeypatch.setenv('V2D_RESOURCE_POLICY','unrestricted')
    def forbidden(*args,**kwargs):raise AssertionError('Admission probe ran despite explicit unrestricted policy')
    monkeypatch.setattr(resource_preflight.Path,'read_text',forbidden)
    monkeypatch.setattr(resource_preflight.subprocess,'run',forbidden)
    resource_preflight.require_reconstruction_capacity()


def test_cold_swap_requires_containment_and_available_ram():
    info=dict(MemAvailable=24,SwapTotal=8,SwapFree=3)
    assert memory_issues(info,{'some':0,'full':0},True)==[]
    assert memory_issues(info,{'some':0,'full':0},False)
    assert memory_issues(dict(info,MemAvailable=15),{},True)
    assert memory_issues(info,{'some':6,'full':0},True)
    assert memory_issues(info,{'some':0,'full':2},True)


def test_lower_ram_admission_requires_enforced_cgroup():
    info=dict(MemAvailable=19.9,SwapTotal=8,SwapFree=5)
    assert memory_issues(info,{},True)==[]
    assert memory_issues(info,{},False)
    assert memory_issues(dict(info,MemAvailable=18),{},True)==[]
    assert memory_issues(dict(info,MemAvailable=17.9),{},True)


def test_missing_silhouette_diagnostic_preserves_candidate_rejection():
    human=np.zeros((32,32),bool);observed=human.copy();observed[8:16,8:16]=True
    metrics=measure_visible_silhouettes(np.zeros((1,32,32),bool),human,observed)
    assert np.isposinf(metrics.centroid_distance[0])
    serialized=serialized_centroid_distance(metrics.centroid_distance[0],metrics.observed_visible_pixels,metrics.rendered_visible_pixels[0])
    assert np.isnan(serialized)
    arrays={key:np.zeros(1,dtype=dtype) for key,dtype in FOUNDATIONPOSE_SELECTION_DTYPES.items()}
    arrays['fp_centroid_distance'][0]=serialized
    validate_foundationpose_selection_arrays(arrays,(1,),'test')
    arrays['fp_centroid_distance'][0]=np.inf
    with pytest.raises(ValueError):validate_foundationpose_selection_arrays(arrays,(1,),'test')


def test_real_nonfinite_errors_still_fail():
    assert serialized_centroid_distance(.3,10,10)==.3
    with pytest.raises(ValueError):serialized_centroid_distance(np.inf,10,10)
    with pytest.raises(ValueError):serialized_centroid_distance(-np.inf,0,0)
    pose=np.eye(4);pose[0,3]=np.inf
    with pytest.raises(ValueError):validate_rigid_transform(pose,'test')
