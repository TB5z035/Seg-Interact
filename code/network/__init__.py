NETWORKS = {}


def register_network(network_name):

    def decorator(cls):
        NETWORKS[network_name] = cls
        return cls

    return decorator


# from . import minkunet
# from . import lightvit
from . import Superpoint_MAE_Pretrain
from . import Superpoint_MAE_Finetune
# from .Superpoint_DN_DETR import Superpoint_DN_DETR