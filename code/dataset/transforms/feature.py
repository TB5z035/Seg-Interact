import numpy as np
import random
import logging
from . import register_transform, Transform


@register_transform('point_chromatic_normalize')
class PointChromaticNormalize(Transform):

    def __call__(self, inputs, labels, extra):
        coords, faces, feats = inputs
        feats = (feats.astype(np.float32) - 128) / 128
        return (coords, faces, feats), labels, extra


@register_transform('point_chromatic_translation')
class PointChromaticTranslation(Transform):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, trans_range_ratio=1e-1, scale=0.3):
        self.trans_range_ratio = trans_range_ratio
        self.scale = scale

    def __call__(self, inputs, labels, extra):  # 0.2
        coords, faces, feats = inputs
        tr = np.clip(np.random.normal(0, self.scale), -1, 1) * 255 * self.trans_range_ratio
        feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return (coords, faces, feats), labels, extra


@register_transform('point_chromatic_auto_contrast')
class ChromaticAutoContrast(Transform):

    def __init__(self, blend_factor=None):
        self.blend_factor = blend_factor

    def __call__(self, inputs, labels, extra):  # 0.2
        coords, faces, feats = inputs
        lo = feats[:, :3].min(0, keepdims=True)
        hi = feats[:, :3].max(0, keepdims=True)
        contrast_feats = (feats[:, :3] - lo) / (hi - lo) * 255
        blend_factor = np.clip(np.random.uniform(0, 1), 0, 1) if self.blend_factor is None else self.blend_factor
        feats[:, :3] = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return (coords, faces, feats), labels, extra


@register_transform('point_chromatic_jitter')
class ChromaticJitter(Transform):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, inputs, labels, extra):  # 0.95
        coords, faces, feats = inputs
        noise = np.random.randn(feats.shape[0], 3)
        noise *= self.std * 255
        feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
        return (coords, faces, feats), labels, extra


@register_transform('hue_saturation_translation')
class HueSaturationTranslation(Transform):

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, inputs, labels, extra):
        coords, faces, feats = inputs
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (2 * random.random() - 1) * self.hue_max
        sat_ratio = 1 + (2 * random.random() - 1) * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        return (coords, faces, feats), labels, extra