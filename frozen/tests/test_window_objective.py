import numpy as np
from scipy.spatial.transform import Rotation
from v2d_syilen.window_objective import temporal_terms,whole_window_score,preconditioners
from v2d_syilen.pruned_tracking import acceleration_cost


def test_uniform_rotation_and_translation_have_no_speed_penalty():
    time=np.arange(50)/30
    r=Rotation.from_rotvec(time[:,None]*np.array([0.,3.,0.])).as_matrix()
    t=time[:,None]*np.array([.2,.1,-.3])
    terms=temporal_terms(r,t,30.)
    assert terms['angular_acceleration']<1e-20 and terms['linear_acceleration']<1e-20


def test_complete_window_matches_causal_sum_and_known_acceleration():
    n=50;fps=30.;time=np.arange(n)/fps
    r=Rotation.from_rotvec((.7*time+.5*1.2*time**2)[:,None]*np.array([0.,0.,1.])).as_matrix()
    accel=np.array([.4,-.1,.2]);t=.5*time[:,None]**2*accel
    ob=np.tile([.2,.15],(n,1));result=whole_window_score(ob,r,t,fps)
    expected=10*.2+20*.15+.01*(n-2)/n*((1.2/(2*np.pi))**2+np.sum(accel**2))
    np.testing.assert_allclose(result['total'],expected,atol=1e-12)
    causal=sum(float(acceleration_cost(r[:i],t[:i],r[i:i+1],t[i:i+1],fps,angular_weight=.01,linear_weight=.01)[0]) for i in range(2,n))/n
    np.testing.assert_allclose(result['total'],5+causal,atol=1e-12)


def test_preconditioners_couple_earlier_and_later_updates_and_are_positive():
    for matrix in preconditioners(50,30.,.01,.01):
        assert np.linalg.eigvalsh(matrix).min()>0
        gradient=np.zeros(50);gradient[25]=1
        update=np.linalg.solve(matrix,gradient)
        assert abs(update[24])>1e-8 and abs(update[26])>1e-8
