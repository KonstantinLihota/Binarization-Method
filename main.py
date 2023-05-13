import argparse
from Binarize import *
from Metrics import Metrics
import os
import imghdr


def Binarize_image():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='path to image directory')
    parser.add_argument('-o', type=str, help='output directory for bin image')
    parser.add_argument('-t', type=str,help='type metod binorize: Niblack, Otsu, Otsu_l, Niblack_multiscale')
    parser.add_argument('--l', type=str, default='None', help='path to directory label')
    parser.add_argument('--m', type=str, default='PSNR', help='metric: PSNR, DRD')
    parser.add_argument('--k', type=int, default=0.2, help='parametr k for method Niblack')
    parser.add_argument('--a', type=int, default=10, help='parametr a for method Niblack')
    parser.add_argument('--w', type=int, default=20, help='window size')
    parser.add_argument('--mw', type=int, default=20, help='parametr min_window for method Niblack_multiscale')
    parser.add_argument('--Mw', type=int, default=1500, help='parametr k for method Niblack_multiscale')
    args = parser.parse_args()
    try:
        files = os.listdir(args.i)
        images = [os.path.join(args.i, file) for file in files if imghdr.what(os.path.join(args.i, file))]
        number = 0
        result_avg = 0
        n = len(images)
        for img_path in images:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if len(image.shape) != 2:
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # cv2.imwrite('/home/kostya/img/analiz' + f'/img{number}.png', image)
                raise ValueError('The picture is a full-color image')
            _, out_path,  method, label_path,metric_type, k, a, window_size, min_window_size, max_window_size = args.__dict__.values()

            if label_path != 'None':
                path_mask_img = args.l + img_path.split('/')[-1]
                mask = cv2.imread(path_mask_img, cv2.IMREAD_GRAYSCALE)
                #cv2.imwrite(img_path.split('/')[-1], mask)

            else:
                mask = None

            bin_image = binarize(image, method, mask, window_size, min_window_size, max_window_size,  k, a)


            cv2.imwrite(out_path + img_path.split('/')[-1], bin_image)
            number += 1

            if mask is not None:
                result = Metrics(metric_type, bin_image, mask)
                print(f"{metric_type} | {img_path.split('/')[-1]} | {result}")
                result_avg+=result/n
        print(f"{metric_type} | all | {result_avg}")

    except Exception as e:
        print(e.args[0])

if __name__ == "__main__":
    Binarize_image()
