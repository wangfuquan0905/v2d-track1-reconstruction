import numpy as np
from scipy.spatial.transform import Rotation
from v2d_syilen.window_search import context_bounds, objective_sum, accepts


def test_local_replacement_delta_includes_every_cross_boundary_triple():
    rng=np.random.default_rng(12);n=80
    r=Rotation.from_rotvec(rng.normal(size=(n,3))*.01).as_matrix()
    t=rng.normal(size=(n,3))*.001;ob=rng.random((n,2))
    for start,stop in [(0,50),(25,75),(30,80),(38,39)]:
        lo,hi=context_bounds(start,stop,n)
        nr=r.copy();nt=t.copy();no=ob.copy()
        nr[start:stop]=Rotation.from_rotvec(rng.normal(size=(stop-start,3))*.02).as_matrix()
        nt[start:stop]+=.02;no[start:stop]*=.8
        global_delta=objective_sum(no,nr,nt,30)-objective_sum(ob,r,t,30)
        local_delta=objective_sum(no[lo:hi],nr[lo:hi],nt[lo:hi],30)-objective_sum(ob[lo:hi],r[lo:hi],t[lo:hi],30)
        np.testing.assert_allclose(local_delta,global_delta,atol=1e-10)


def test_rejects_worse_equal_and_nonfinite_candidates():
    assert accepts(3.,2.9)
    assert not any(accepts(3.,v) for v in [3.,4.,np.nan,np.inf])
