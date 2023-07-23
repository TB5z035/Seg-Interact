import torch

OPTIMIZERS = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
}

SCHEDULERS = {
    None: None,
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}
