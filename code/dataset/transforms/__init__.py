import numpy as np
import logging

logger = logging.getLogger('transform_geometry')

TRANSFORMS = {}


def register_transform(transform_name):

    def decorator(cls):
        TRANSFORMS[transform_name] = cls
        return cls

    return decorator


class Transform:

    def __call__(self, coords, faces, feats, labels, maps):
        raise NotImplementedError


class Compose(Transform):

    def __init__(self, *args) -> None:
        self.args = args

    def __call__(self, coords, faces, feats, labels, extra):
        for t in self.args:
            logger.info(t)
            coords, faces, feats, labels, extra = t(coords, faces, feats, labels, extra)
        return coords, faces, feats, labels, extra


@register_transform('random_apply')
class RandomApply(Transform):

    def __init__(self, prob, inner_t, **kwargs) -> None:
        self.inner_t = TRANSFORMS[inner_t](**kwargs)
        self.prob = prob

    def __call__(self, coords, faces, feats, labels, extra) -> None:
        if np.random.random() < self.prob:
            coords, faces, feats, labels, extra = self.inner_t(coords, faces, feats, labels, extra)
        return coords, faces, feats, labels, extra

def parse_transform(transforms: list):
    return Compose(*[TRANSFORMS[list(t.keys())[0]](**(list(t.values())[0] if list(t.values())[0] else {})) for t in transforms])

from . import geometry
from . import feature
