import itertools
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from v2d_syilen.rotation_temporal import angular_derivatives
from v2d_syilen.orientation_search import so3_path


def test_known_angular_velocity_acceleration_and_frame_invariance():
    fps = 30.
    time = torch.arange(12, dtype=torch.float64)/fps
    theta = .7*time + .5*1.2*time.square()
    rv = torch.stack([theta*0, theta*0, theta], -1)
    rotations = axis_angle_to_matrix(rv)
    velocity, acceleration = angular_derivatives(rotations, fps)
    torch.testing.assert_close(velocity[:, 2], .7+1.2*(time[1:]+time[:-1])/2)
    torch.testing.assert_close(acceleration[:, 2], torch.full_like(time[2:], 1.2))
    basis = axis_angle_to_matrix(torch.tensor([.3, -.2, .5], dtype=torch.float64))
    v2, a2 = angular_derivatives(basis@rotations, fps)
    torch.testing.assert_close(v2, velocity@basis.T)
    torch.testing.assert_close(a2, acceleration@basis.T)
    constant = axis_angle_to_matrix(torch.stack([time*0, time*0, .7*time], -1))
    _, alpha = angular_derivatives(constant, fps)
    torch.testing.assert_close(alpha, torch.zeros_like(alpha), atol=1e-10, rtol=0)


def test_zero_and_near_pi_gradients_are_finite():
    for angle in [0., 1e-8, np.pi-1e-4]:
        rv = torch.tensor([[0., 0., 0.], [0., angle, 0.], [0., angle, 0.]], requires_grad=True, dtype=torch.float64)
        omega, alpha = angular_derivatives(axis_angle_to_matrix(rv), 30.)
        loss = omega.square().sum()+alpha.square().sum()
        loss.backward()
        assert torch.isfinite(rv.grad).all()
    rv = torch.randn(5, 3, dtype=torch.float64)*.1
    rv.requires_grad_()
    assert torch.autograd.gradcheck(lambda x: angular_derivatives(axis_angle_to_matrix(x), 30.), (rv,))


def test_halo_batches_match_full_sequence_values_and_gradients():
    torch.manual_seed(12)
    rv = (torch.randn(13, 3, dtype=torch.float64)*.03).requires_grad_()
    r = axis_angle_to_matrix(rv)
    omega, alpha = angular_derivatives(r, 30.)
    full = (omega/(np.pi/2)).square().sum(-1).mean()+(alpha/(2*np.pi)).square().sum(-1).mean()
    chunked = 0
    n = len(r)
    for begin in range(0, n, 4):
        end = min(n, begin+4)
        left, right = max(0, begin-1), min(n, end+1)
        w, a = angular_derivatives(r[left:right], 30.)
        edges = torch.arange(begin, min(end, n-1))-left
        centers = torch.arange(max(begin, 1), min(end, n-1))-left
        chunked += (w[edges]/(np.pi/2)).square().sum()/(n-1)+(a[centers-1]/(2*np.pi)).square().sum()/(n-2)
    torch.testing.assert_close(chunked, full)
    g1 = torch.autograd.grad(full, rv, retain_graph=True)[0]
    g2 = torch.autograd.grad(chunked, rv)[0]
    torch.testing.assert_close(g1, g2)


def test_so3_search_matches_exhaustive_continuous_objective():
    rng = np.random.default_rng(9)
    n, k = 4, 3
    rot = axis_angle_to_matrix(torch.tensor(rng.normal(0, .1, (n, k, 3)), dtype=torch.float64))
    trans = rng.normal(0, .001, (n, k, 3))
    unary = rng.uniform(0, 20, (n, k))
    def score(path):
        w, a = angular_derivatives(rot[np.arange(n), path], 30.)
        return (unary[np.arange(n), path].mean()+float((w/(np.pi/2)).square().sum(-1).mean())
                +float((a/(2*np.pi)).square().sum(-1).mean())
                +.1*np.square(np.diff(trans[np.arange(n), path], n=2, axis=0)/.005).mean())
    path, value = so3_path(unary, rot.numpy(), trans, 30.)
    brute = min(score(p) for p in itertools.product(range(k), repeat=n))
    np.testing.assert_allclose(value, brute)
    np.testing.assert_allclose(score(path), brute)

    def separate_linear_score(path):
        t = trans[np.arange(n), path]
        old_linear = .1*np.square(np.diff(t,n=2,axis=0)/.005).mean()
        return (score(path)-old_linear+.1*np.square(np.diff(t,axis=0)*30/.5).sum(-1).mean()
                +np.square(np.diff(t,n=2,axis=0)*30**2).sum(-1).mean())
    path,value = so3_path(unary,rot.numpy(),trans,30.,linear_velocity_weight=.1,linear_acceleration_weight=1.)
    brute = min(separate_linear_score(p) for p in itertools.product(range(k),repeat=n))
    np.testing.assert_allclose(value,brute)
    np.testing.assert_allclose(separate_linear_score(path),brute)
