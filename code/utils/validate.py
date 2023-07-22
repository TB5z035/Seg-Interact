import MinkowskiEngine as ME
import torch
from torch.utils.data import DataLoader
import logging

from .misc import get_device, to_device

device = get_device()
logger = logging.getLogger('validate')


def validate(model, val_loader: DataLoader, criterion, metrics=[]):
    model.eval()
    with torch.no_grad():
        metric_objs = [metric(val_loader.dataset.num_train_classes, val_loader.dataset.ignore_class) for metric in metrics]
        loss_sum = 0
        for idx, (inputs, labels, maps) in enumerate(val_loader):
            output = model(to_device(inputs, device))
            loss_sum += criterion(output, to_device(labels, device)).item()

            pred = output.argmax(dim=1).cpu()
            # FIXME handle batch size > 1
            for metric in metric_objs:
                if maps is None:
                    metric.record(pred, labels)
                else:
                    metric.record(pred[maps[1]], labels[maps[1]])
            logging.info(f'progress: {idx}/{len(val_loader)}')

    metric_results = {metric.NAME: metric.calc() for metric in metric_objs}
    for metric in metric_objs:
        logging.info(f'{metric.NAME}: {metric_results[metric.NAME]}')
    return loss_sum / len(val_loader), metric_results