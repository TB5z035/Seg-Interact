import MinkowskiEngine as ME
import torch
from torch.utils.data import DataLoader
import logging

from .misc import get_device, to_device, get_local_rank

device = get_device()
logger = logging.getLogger('validate')

from tqdm import tqdm


def validate(model, val_loader: DataLoader, criterion, metrics=[], writer=None, global_iter=None):
    if get_local_rank() == 0:
        model.eval()
        with torch.no_grad():
            metric_objs = [
                metric(
                    val_loader.dataset.num_train_classes,
                    val_loader.dataset.ignore_class,
                    val_loader.dataset.train_class_names,
                ) for metric in metrics
            ]
            loss_sum = 0
            for idx, (inputs, labels, maps) in enumerate(tqdm(val_loader)):
                output = model(to_device(inputs, device))
                loss_sum += criterion(output, to_device(labels, device)).item()

                pred = output.argmax(dim=1).cpu()
                # FIXME handle batch size > 1
                for metric in metric_objs:
                    if maps is None:
                        metric.record(pred, labels)
                    else:
                        metric.record(pred[maps[1]], labels[maps[1]])
                # logging.info(f'progress: {idx}/{len(val_loader)}')

        metric_results = {metric.NAME: metric.calc() for metric in metric_objs}
        for metric in metric_objs:
            metric.log(logger, writer=writer, global_iter=global_iter, name_prefix='val/')
        return loss_sum / len(val_loader), metric_results
    else:
        return 0, {}