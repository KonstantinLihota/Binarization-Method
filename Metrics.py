from enum import Enum
import cv2


class MetricsMethod(Enum):
    DRD = 'DRD'
    PSNR = 'PSNR'


def PSNR(image, label):
    return cv2.PSNR(image, label)


class Metrics:
    def __init__(self, metric_name):
        self.metric = cv2.PSNR
        if metric_name == MetricsMethod.PSNR:
            self.metric = PSNR
        # elif metric_name == MetricsMethod.DRD:

    def get_metric(self):
        return self.metric

    def slove_metric(self, image, label):
        return self.metric(image, label)
