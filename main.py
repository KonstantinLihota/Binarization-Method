import argparse
import cv2
from Binarize import *
from Metrics import Metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='path to image')
    parser.add_argument('-o', type=str, help='output dir for bin image')

    parser.add_argument('--l', type=str, default='None', help='path to label image')
    parser.add_argument('--t', type=str, default='Otsu',
                        help='type metod binorize: Niblack, Otsu, Otsu_l, Niblack_multiscale')
    parser.add_argument('--m', type=str, default='PSNR', help='metric: PSNR, DRD')
    parser.add_argument('--k', type=int, default=0.2, help='parametr k for method Niblack')
    parser.add_argument('--a', type=int, default=10, help='parametr a for method Niblack')
    parser.add_argument('--w', type=int, default=50, help='window size')
    parser.add_argument('--mw', type=int, default=2, help='parametr min_window for method Niblack_multiscale')
    parser.add_argument('--Mw', type=int, default=50, help='parametr k for method Niblack_multiscale')
    args = parser.parse_args()
    try:
        image = cv2.imread(args.i)
        if len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY)
            # raise ValueError('The picture is a full-color image')

        _, out_path, label_path, method, metric_type, k, a, window_size, min_window_size, max_window_size = args
        if label_path != 'None':
            mask = cv2.imread(args.l)
        else:
            mask = None
        metric = Metrics(metric_type)
        bin_image = binarize(image, method, mask, window_size, min_window_size, max_window_size, metric, k, a)
        cv2.imwrite(out_path, bin_image)

        if mask is not None:
            result = metric(image, mask)
            print(result)

    except Exception as e:
        print(e.args[0])
