import numpy as np

POLICIES = {}


def register_policy(policy_name):

    def decorator(cls):
        POLICIES[policy_name] = cls
        return cls

    return decorator


class SelectPolicyBase:

    @staticmethod
    def select(data_arr: np.ndarray, label_arr: np.ndarray, *args, **kwargs):
        raise NotImplementedError


from . import mixture_filter