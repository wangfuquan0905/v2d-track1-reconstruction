"""Object observation loss must not reward hiding or displacing geometry."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('nvdiffrast.torch')
from v2d_syilen.render_observation import observation_terms


def test_object_color_penalizes_missing_and_displaced_foreground():
    rgb = torch.full((1, 8, 8, 3), .6)
    masks = torch.zeros(1, 8, 8, 2)
    masks[:, 2:6, 2:6, 1] = 1
    perfect = torch.zeros(1, 8, 8, 5)
    perfect[..., 1] = masks[..., 1]
    perfect[..., 2:] = rgb*masks[..., 1:2]
    loss = lambda pred: observation_terms(pred, rgb, masks)['object_color']
    assert loss(perfect).item() == 0
    absent = loss(torch.zeros_like(perfect)).item()
    assert absent == pytest.approx(.6)
    assert loss(torch.roll(perfect, 4, dims=2)).item() > absent
    coverage = torch.tensor(.5, requires_grad=True)
    loss(perfect*coverage).sum().backward()
    assert coverage.grad.item() < 0  # Gradient descent increases coverage.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA rasterizer required')
def test_numerical_translation_update_reduces_rerendered_observation_error():
    from v2d_syilen.render_observation import ObservationRenderer, translation_observation_gradient
    device = 'cuda'
    vertices = torch.tensor([[-.4,-.4,2.],[.4,-.4,2.],[.4,.4,2.],[-.4,.4,2.]], device=device)
    faces = torch.tensor([[0,1,2],[0,2,3]], dtype=torch.int32, device=device)
    colors = torch.tensor([[.7,.3,.1]]*4, device=device)
    renderer = ObservationRenderer(torch.empty(0,3,dtype=torch.int32,device=device), faces, colors,
        torch.tensor([64.,64.,32.,32.],device=device),64,64,supersample=4)
    human = torch.empty(1,0,3,device=device)
    target = renderer(human,vertices[None])
    masks = target[...,:2].detach()
    rgb = colors[0].expand(1,64,64,3)
    predicted = vertices[None]+torch.tensor([.1,-.08,0.],device=device)
    weights = dict(human_silhouette=0.,human_area=0.,object_area=0.,object_color=20.,object_silhouette=10.)
    def loss(obj):
        terms = observation_terms(renderer(human,obj),rgb,masks)
        return sum(weights[k]*value for k,value in terms.items())
    g = translation_observation_gradient(renderer,human,predicted,rgb,masks,weights)
    assert g[0,0] > 0 and g[0,1] < 0
    updated = predicted-.002*g[:,None]/g.norm(dim=-1)[:,None,None]
    assert loss(updated).item() < loss(predicted).item()
