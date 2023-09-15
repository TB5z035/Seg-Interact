import sys
import torch_geometric.transforms as pygT
from omegaconf import OmegaConf
from .transforms import *
from .sampling import *
from .data import *
from .point import *
from .geometry import *
from .neighbors import *
from .partition import *
from .device import *
from .graph import *
from ...data import Data

_spt_tr = sys.modules[__name__]
_pyg_tr = sys.modules["torch_geometric.transforms"]

_intersection_tr = set(_spt_tr.__dict__) & set(_pyg_tr.__dict__)
_intersection_tr = set([t for t in _intersection_tr if not t.startswith("_")])
_intersection_cls = []

for name in _intersection_tr:
    cls = getattr(_spt_tr, name)
    if not "torch_geometric.transforms." in str(cls):
        _intersection_cls.append(cls)

if len(_intersection_tr) > 0:
    if len(_intersection_cls) > 0:
        raise Exception(f"It seems that you are overriding a transform from pytorch "
                        f"geometric, this is forbidden, please rename your classes "
                        f"{_intersection_tr} from {_intersection_cls}")
    else:
        raise Exception(f"It seems you are importing transforms {_intersection_tr} "
                        f"from pytorch geometric within the current code base. Please, "
                        f"remove them or add them within a class, function, etc.")


def instantiate_transform(transform_option, attr="transform"):
    """Create a transform from an OmegaConf dict such as

    ```
    transform: GridSampling3D
        params:
            size: 0.01
    ```
    """
    # Read the transform name
    tr_name = getattr(transform_option, attr, None)

    # Find the transform class corresponding to the name
    cls = getattr(_spt_tr, tr_name, None)
    if not cls:
        cls = getattr(_pyg_tr, tr_name, None)
        if not cls:
            raise ValueError(f"Transform {tr_name} is nowhere to be found")

    # Parse the transform arguments
    try:
        tr_params = transform_option.get('params')  # Update to OmegaConf 2.0
        if tr_params is not None:
            tr_params = OmegaConf.to_container(tr_params, resolve=True)
    except KeyError:
        tr_params = None
    try:
        lparams = transform_option.get('lparams')  # Update to OmegaConf 2.0
        if lparams is not None:
            lparams = OmegaConf.to_container(lparams, resolve=True)
    except KeyError:
        lparams = None

    # Instantiate the transform
    if tr_params and lparams:
        return cls(*lparams, **tr_params)
    if tr_params:
        return cls(**tr_params)
    if lparams:
        return cls(*lparams)
    return cls()


def instantiate_transforms(transform_options):
    """Create a torch_geometric composite transform from an OmegaConf
    list such as

    ```
    - transform: GridSampling3D
        params:
            size: 0.01
    - transform: NormaliseScale
    ```
    """
    transforms = []
    for transform in transform_options:
        transforms.append(instantiate_transform(transform))

    if len(transforms) <= 1:
        return pygT.Compose(transforms)

    # If multiple transforms are composed, make sure the input and
    # output match
    for i in range(1, len(transforms)):
        t_out = transforms[i - 1]
        t_in = transforms[i]
        out_type = getattr(t_out, '_OUT_TYPE', Data)
        in_type = getattr(t_in, '_IN_TYPE', Data)
        if in_type != out_type:
            raise ValueError(f"Cannot compose transforms: {t_out} returns a {out_type} "
                             f"while {t_in} expects a {in_type} input.")

    return pygT.Compose(transforms)
