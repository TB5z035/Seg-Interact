import MinkowskiEngine as ME
import torch
from torch.utils.data import DataLoader

from .misc import get_device

device = get_device()


def validate(model, val_loader: DataLoader, criterion, metrics=[]):
    model.eval()
    with torch.no_grad():
        metric_objs = [metric(val_loader.dataset.num_train_classes) for metric in metrics]
        loss_sum = 0
        for _, (coords, feats, _, labels, maps) in enumerate(val_loader):
            coords, feats, labels = coords.to(device), feats.to(device), labels.to(device)
            input = ME.SparseTensor(feats, coordinates=coords, device=device)
            label = labels.to(device)

            output = model(input)

            loss = criterion(output.F, label).item()
            loss_sum += loss

            pred = output.F.argmax(dim=1)
            for metric in metric_objs:
                if maps is None:
                    metric.record(pred, label)
                else:
                    metric.record(pred[maps[1]], label[maps[1]])

    metric_results = {metric.NAME: metric.calc() for metric in metric_objs}
    return loss_sum / len(val_loader), metric_results