DATASETS = {}


def register_dataset(dataset_name):

    def decorator(cls):
        DATASETS[dataset_name] = cls
        return cls

    return decorator


from . import scannet
from . import multiview_classification