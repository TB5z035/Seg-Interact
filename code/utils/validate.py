import torch
from torch.utils.data import DataLoader
import logging

from .misc import get_device, to_device, get_local_rank

device = get_device()
logger = logging.getLogger('validate')

from tqdm import tqdm


def validate(model, val_loader: DataLoader, criterion=None, metrics=[], writer=None, global_iter=None):
    if get_local_rank() == 0:
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            metric_objs = [
                metric(
                    val_loader.dataset.num_train_classes,
                    val_loader.dataset.ignore_class,
                    val_loader.dataset.train_class_names,
                ) for metric in metrics
            ]

            for idx, (inputs, labels, extras) in enumerate(tqdm(val_loader)):
                output = model(to_device(inputs, device))
                pred = output.argmax(dim=1).cpu()
                if 'maps' in extras:
                    output = output[extras['maps'][1]]
                    pred = pred[extras['maps'][1]]
                    labels = labels[extras['maps'][1]]

                if criterion is not None:
                    loss_sum += criterion(output, to_device(labels, device)).item()

                for metric in metric_objs:
                    metric.record(pred, labels)

        loss_sum /= len(val_loader)

        metric_results = {}
        for metric in metric_objs:
            metric_results[metric.NAME] = metric.calc()
            metric.log(logger, writer=writer, global_iter=global_iter, name_prefix='val/')

        return loss_sum, metric_results
    else:
        return 0, {}


if __name__ == '__main__':
    pass
