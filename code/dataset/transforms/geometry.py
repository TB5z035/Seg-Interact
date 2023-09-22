import numpy as np
import random
import logging
import scipy
from . import register_transform, Transform

logger = logging.getLogger('transform_geometry')

translation_mat = lambda dx, dy, dz: np.array(
    (
        (1, 0, 0, dx),
        (0, 1, 0, dy),
        (0, 0, 1, dz),
        (0, 0, 0, 1),
    ), dtype=np.float64)


def rotation_mat(x, y, z) -> np.ndarray:
    along_z = np.array((
        (np.cos(z), -np.sin(z), 0, 0),
        (np.sin(z), np.cos(z), 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ),
                       dtype=np.float64)
    along_y = np.array((
        (np.cos(y), 0, -np.sin(y), 0),
        (0, 1, 0, 0),
        (np.sin(y), 0, np.cos(y), 0),
        (0, 0, 0, 1),
    ),
                       dtype=np.float64)
    along_x = np.array((
        (1, 0, 0, 0),
        (0, np.cos(x), -np.sin(x), 0),
        (0, np.sin(x), np.cos(x), 0),
        (0, 0, 0, 1),
    ),
                       dtype=np.float64)

    return along_z @ along_y @ along_x


@register_transform('point_to_center')
class PointToCenter(Transform):

    def __call__(self, inputs, labels, extra: dict):
        coords, faces, feats = inputs
        return (coords - coords.mean(0), faces, feats), labels, extra


@register_transform('point_to_positive')
class PointToPositive(Transform):

    def __call__(self, inputs, labels, extra: dict):
        coords, faces, feats = inputs
        return (coords - coords.min(0), faces, feats), labels, extra


@register_transform('random_rotation')
class RandomRotation(Transform):

    def __init__(self, angle_v=3, angle_h=180, upright_axis='z') -> None:
        angle_v = angle_v / 180 * np.pi
        angle_h = angle_h / 180 * np.pi
        self.angle_h = angle_h
        self.angle_v = angle_v
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]

    def __call__(self, inputs, labels, extra: dict):
        coords, faces, feats = inputs
        center = coords - coords.mean(0)
        angle = [np.random.uniform(-self.angle_v, self.angle_v) for _ in range(3)]
        angle[self.upright_axis] = np.random.uniform(-self.angle_h, self.angle_h)
        rot_mat = rotation_mat(*angle)[:3, :3]
        coords = (coords - center) @ rot_mat + center

        return (coords, faces, feats), labels, extra


@register_transform('random_scale')
class RandomScale(Transform):

    def __init__(self, scale_rng=(0.9, 1.1)) -> None:
        self.scale_rng = scale_rng

    def __call__(self, inputs, labels, extra: dict):
        coords, faces, feats = inputs
        scale = np.random.uniform(*self.scale_rng)
        scale = coords * scale
        return (coords, faces, feats), labels, extra


@register_transform('random_dropout')
class RandomDropout(Transform):
    """
    This transform invalidate the unique_map and inverse_map, thus should not be used during validation / testing.
    """

    def __init__(self, dropout_ratio=0.2):
        self.dropout_ratio = dropout_ratio

    def __call__(self, inputs, labels, extra: dict):
        coords, faces, feats = inputs
        N = coords.shape[0]
        inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
        if 'full_super_indices_10' in extra.keys():
            extra['linearity'] = extra['linearity'][inds]
            extra['planarity'] = extra['planarity'][inds]
            extra['scattering'] = extra['scattering'][inds]
            extra['verticality'] = extra['verticality'][inds]
            extra['elevation'] = extra['elevation'][inds]
            extra['full_super_indices_10'] = extra['full_super_indices_10'][inds]
        return (coords[inds], faces, feats[inds]), labels[inds], extra


@register_transform('random_horizontal_flip')
class RandomHorizontalFlip(Transform):

    def __init__(self, upright_axis):
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        self.horizontal_axes = [i for i in range(3) if i != self.upright_axis]

    def __call__(self, inputs, labels, extra: dict):
        coords, faces, feats = inputs
        for curr_ax in self.horizontal_axes:
            if random.random() < 0.5:
                coord_mean = np.mean(coords[:, curr_ax])
                coords[:, curr_ax] = 2 * coord_mean - coords[:, curr_ax]
        return (coords, faces, feats), labels, extra


@register_transform('random_translation')
class RandomTranslation(Transform):

    def __init__(self, translation_range=0.2):
        self.translation_range = translation_range

    def __call__(self, inputs, labels, extra: dict):
        coords, faces, feats = inputs
        translation = np.random.uniform(-self.translation_range, self.translation_range, size=(3,))
        return (coords + translation, faces, feats), labels, extra


@register_transform('point_cloud_size_limit')
class PointCloudSizeLimit(Transform):
    """
    This transform invalidate the unique_map and inverse_map, thus should not be used during validation / testing.
    FIXME: the faces are not updated
    """

    def __init__(self, max_num=200000):
        self.max_num = max_num

    def __call__(self, inputs, labels, extra: dict):
        coords, faces, feats = inputs
        N = coords.shape[0]
        if self.max_num > N:
            return (coords, faces, feats), labels, extra
        if self.max_num / N < 1:
            logger.debug(f"Pointcloud constraint {self.max_num} too high for {N}-point point cloud")
        inds = np.random.choice(N, self.max_num, replace=False)
        if 'full_super_indices_10' in extra.keys():
            extra['linearity'] = extra['linearity'][inds]
            extra['planarity'] = extra['planarity'][inds]
            extra['scattering'] = extra['scattering'][inds]
            extra['verticality'] = extra['verticality'][inds]
            extra['elevation'] = extra['elevation'][inds]
            extra['full_super_indices_10'] = extra['full_super_indices_10'][inds]
        return (coords[inds], faces, feats[inds]), labels[inds], extra


@register_transform('elastic_distortion')
class ElasticDistortion(Transform):

    def __init__(self, distortion_params):
        self.distortion_params = distortion_params

    def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

          pointcloud: numpy array of (number of points, at least 3 spatial dims)
          granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
          magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords += interp(coords) * magnitude
        return coords, feats, labels

    def __call__(self, inputs, labels, extra: dict):  # 0.95
        coords, faces, feats = inputs
        for granularity, magnitude in self.distortion_params:
            coords, feats, labels = self.elastic_distortion(coords, feats, labels, granularity, magnitude)
        return (coords, faces, feats), labels, extra
