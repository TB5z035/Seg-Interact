import numpy as np

METRICS = {}


def register_metric(metric_name):

    def decorator(cls):
        METRICS[metric_name] = cls
        return cls

    return decorator


from . import metrics
