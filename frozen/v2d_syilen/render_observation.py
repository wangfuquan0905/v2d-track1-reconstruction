"""Differentiable camera-space scene rendering and RGB observation losses.

Both a simulated trajectory and a directly optimized trajectory can be passed
to this renderer. It does not itself run dynamics or use reference 3D data.
"""
import torch
import nvdiffrast.torch as dr


class ObservationRenderer:
    def __init__(self, human_faces, object_faces, object_colors, K, height, width, supersample=1):
        self.context = dr.RasterizeCudaContext()
        self.human_faces = human_faces.int().contiguous()
        self.object_faces = object_faces.int().contiguous()
        self.object_colors = object_colors
        self.K, self.height, self.width = K, height, width
        if not isinstance(supersample, int) or supersample < 1:
            raise ValueError('supersample must be a positive integer')
        self.supersample = supersample

    def __call__(self, human, obj):
        """Return visible person/object alpha and premultiplied object RGB.

        One depth buffer resolves person/object occlusion. OpenCV pixel y is
        mapped to increasing raster rows; no post-render image flip is used.
        Antialiasing supplies silhouette gradients at triangle boundaries.
        """
        vertices = torch.cat([human, obj], dim=1)
        faces = torch.cat([self.human_faces, self.object_faces + human.shape[1]], dim=0)
        x, y, z = vertices.unbind(-1)
        fx, fy, cx, cy = self.K
        near, far = .01, 100.
        clip = torch.stack([2*fx/self.width*x+(2*cx/self.width-1)*z,
                            2*fy/self.height*y+(2*cy/self.height-1)*z,
                            (far+near)/(far-near)*z-2*far*near/(far-near), z], -1).contiguous()
        raster, _ = dr.rasterize(self.context, clip, faces, resolution=[self.height*self.supersample, self.width*self.supersample])
        attributes = torch.zeros((vertices.shape[1], 5), device=vertices.device)
        attributes[:human.shape[1], 0] = 1
        attributes[human.shape[1]:, 1] = 1
        attributes[human.shape[1]:, 2:] = self.object_colors
        rendered, _ = dr.interpolate(attributes[None].expand(len(vertices), -1, -1).contiguous(), raster, faces)
        rendered = dr.antialias(rendered, raster, clip, faces).clamp(0, 1)
        if self.supersample > 1:
            rendered = torch.nn.functional.interpolate(rendered.permute(0, 3, 1, 2),
                size=(self.height, self.width), mode='area').permute(0, 2, 3, 1).contiguous()
        return rendered


def observation_terms(rendered, rgb, masks):
    """Per-frame losses: visible-mask soft IoU, normalized size, object color.

    Masks are RGB-derived person/object segmentation, never multiview GT.
    Human color is absent. Object RGB uses fixed mesh vertex colors, without
    trainable appearance or lighting, to prevent explaining pose via recoloring.
    """
    alpha = rendered[..., :2]
    intersection = (alpha*masks).sum((1, 2))
    union = (alpha+masks-alpha*masks).sum((1, 2)).clamp_min(1)
    silhouette = 1-intersection/union
    target_area = masks.sum((1, 2)).clamp_min(1)
    area = ((alpha.sum((1, 2))-target_area)/target_area).square()
    # Compare foreground images using the FIXED observed support. Multiplying
    # the target by predicted alpha would let disappearing/misaligned geometry
    # erase its own color residual. Include spill outside observed support too.
    target_object_rgb = rgb*masks[..., 1:2]
    color = (rendered[..., 2:]-target_object_rgb).abs().sum((1, 2, 3)) / (3*target_area[:, 1])
    return {'human_silhouette': silhouette[:, 0], 'object_silhouette': silhouette[:, 1],
            'human_area': area[:, 0], 'object_area': area[:, 1], 'object_color': color}


@torch.no_grad()
def translation_observation_gradient(renderer, human, obj, rgb, masks, weights, epsilon=.002):
    """Central differences of the same rendered objective, per frame in metres.

    Only the rigid object's camera translation is perturbed. This replaces
    unreliable raster coverage derivatives for translation; it is not a
    gradient rescaling or an extra loss. Other derivatives remain autograd.
    """
    if epsilon <= 0:
        raise ValueError('Translation finite-difference step must be positive')
    human, obj, rgb, masks = (x.detach() for x in (human, obj, rgb, masks))
    batch = len(obj)
    columns = []
    for axis in range(3):
        offset = torch.zeros(3, device=obj.device, dtype=obj.dtype)
        offset[axis] = epsilon
        rendered = renderer(torch.cat([human, human]), torch.cat([obj+offset, obj-offset]))
        terms = observation_terms(rendered, torch.cat([rgb, rgb]), torch.cat([masks, masks]))
        loss = sum(weights[key]*value for key, value in terms.items())
        columns.append((loss[:batch]-loss[batch:])/(2*epsilon))
    result = torch.stack(columns, -1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite translation image derivative')
    return result
