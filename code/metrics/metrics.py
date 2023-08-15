from . import register_metric
import numpy as np
import torch


class BaseMetric:

    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, pred, target):
        pass

    def calc(self):
        pass


# Maybe base class is wrong
@register_metric('IoU')
class IoU(BaseMetric):
    NAME = 'IoU'

    def __init__(self, num_class, ignore_label, class_names, *args, **kwargs) -> None:
        self.num_class = num_class
        self.hist_stat = np.zeros((num_class, num_class), dtype=np.int64)
        self.ignore_label = ignore_label
        self.class_names = class_names

    def record(self, pred, target):
        """
        Args:
            pred: numpy.ndarray, shape: (B, N), int32
            target: numpy.ndarray, shape: (B, N), int32
            ignore_label: int32
        """
        ignore_label = self.ignore_label
        flatten_pred = pred.flatten()
        flatten_target = target.flatten()
        mask = (flatten_target != ignore_label) & (flatten_pred != ignore_label)
        self.hist_stat += np.bincount(flatten_pred[mask] + flatten_target[mask] * self.num_class,
                                      minlength=self.num_class**2).reshape(self.num_class, self.num_class)

    def calc(self):
        ious = np.diag(self.hist_stat) / (self.hist_stat.sum(0) + self.hist_stat.sum(1) - np.diag(self.hist_stat))
        return ious

    def log(self, logger, writer=None, global_iter=None, name_prefix=''):
        ious = self.calc()
        for idx, iou in enumerate(ious):
            logger.info(f'{self.class_names[idx]:10}: {iou:.4f}')
            if writer is not None:
                writer.add_scalar(f'{name_prefix}{self.NAME}/{self.class_names[idx]}', iou, global_iter)


@register_metric('mIoU')
class mIoU(IoU):
    NAME = 'mIoU'

    def calc(self):
        ious = super().calc()
        return np.nanmean(ious)

    def log(self, logger, writer=None, global_iter=None, name_prefix=''):
        iou = self.calc()
        logger.info(f'{self.NAME}: {iou:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}', iou, global_iter)


@register_metric("Acc")
class Acc(BaseMetric):
    NAME = 'Acc'

    def __init__(self, num_class, ignore_label, class_names, *args, **kwargs) -> None:
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.hist_stat = np.zeros((num_class, num_class), dtype=np.int64)
        self.class_names = class_names

    def record(self, pred, target):
        """
        Args:
            pred: numpy.ndarray, shape: (B, 1), int32
            target: numpy.ndarray, shape: (B, 1), int32
            ignore_label: int32
        """
        ignore_label = self.ignore_label
        flatten_pred = pred.flatten()
        flatten_target = target.flatten()
        mask = (flatten_target != ignore_label) & (flatten_pred != ignore_label)
        self.hist_stat += np.bincount(flatten_pred[mask] + flatten_target[mask] * self.num_class,
                                      minlength=self.num_class**2).reshape(self.num_class, self.num_class)

    def calc(self):
        correct = np.diag(self.hist_stat).sum()
        total = self.hist_stat.sum()
        accuracy = correct / total
        return accuracy

    def log(self, logger, writer=None, global_iter=None, name_prefix=''):
        accuracy = self.calc()
        logger.info(f'Overall Accuracy: {accuracy:.4f}')
        if writer is not None:
            writer.add_scalar(f'{name_prefix}{self.NAME}/Overall', accuracy, global_iter)
