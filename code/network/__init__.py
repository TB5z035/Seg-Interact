NETWORKS = {}


def register_network(network_name):

    def decorator(cls):
        NETWORKS[network_name] = cls
        return cls

    return decorator

# from . import minkunet
from . import lightvit