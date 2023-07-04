import numpy as np


class SegmentMetric:

    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, pred, target):
        pass

    def calc(self):
        pass


class IoU(SegmentMetric):
    NAME = 'IoU'

    def __init__(self, num_class, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_class = num_class
        self.hist_stat = np.zeros((num_class, num_class), dtype=np.int64)

    def record(self, pred, target, ignore_label=None):
        """
        Args:
            pred: numpy.ndarray, shape: (B, N), int32
            target: numpy.ndarray, shape: (B, N), int32
            ignore_label: int32
        """
        flatten_pred = pred.flatten()
        flatten_target = target.flatten()
        mask = (flatten_target != ignore_label) & (flatten_pred != ignore_label)
        self.hist_stat += np.bincount(flatten_pred[mask] + flatten_target[mask] * self.num_class,
                                      minlength=self.num_class**2).reshape(self.num_class, self.num_class)

    def calc(self):
        ious = np.diag(self.hist_stat) / (self.hist_stat.sum(0) + self.hist_stat.sum(1) - np.diag(self.hist_stat))
        return ious


class mIoU(SegmentMetric):
    NAME = 'mIoU'

    def calc(self):
        ious = super().calc()
        return ious.nanmean().item()
