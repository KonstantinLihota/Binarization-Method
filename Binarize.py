from enum import Enum
import numpy as np
from scipy.stats import norm
import cv2


class BinarizationMethod(Enum):
    NIBLACK = 'Niblack'
    OTSU = 'Otsu'
    OTSU_L = 'Otsu_l'
    NIBLACK_MULTISCALE = 'Niblack_multiscale'


def binarize(image, method, mask, window_size, min_window_size, max_window_size, metric, k, a):
    if method == BinarizationMethod.NIBLACK:
        bin_image = niblack_threshold(image, window_size, k, a)
    elif method == BinarizationMethod.OTSU:
        bin_image = otsu_thresholding(image)
    elif method == BinarizationMethod.OTSU_L:
        bin_image = maximum_likelihood_thresholding(image)
    elif method == BinarizationMethod.NIBLACK_MULTISCALE:
        bin_image = adaptive_niblack_threshold(image, mask, min_window_size, max_window_size, metric, k, a)
    else:
        raise ValueError('Unknown binarization method')
    return bin_image

def otsu_thresholding(image):

    hist, bins = np.histogram(image, bins=256, range=(0, 255))
    n_pixels = np.sum(hist)

    norm_hist = hist / n_pixels

    weights = np.cumsum(norm_hist)
    means = np.cumsum(np.arange(256) * norm_hist)

    global_mean = means[-1]

    variances = ((global_mean * weights - means)**2) / (weights * (1 - weights))

    threshold = np.argmax(variances)

    return threshold

def maximum_likelihood_thresholding(image):

    hist, bins = np.histogram(image, bins=256, range=(0, 255))
    n_pixels = np.sum(hist)
    group1 = hist[:128]
    group2 = hist[128:]

    mean1 = np.sum(np.arange(128) * group1) / np.sum(group1)
    mean2 = np.sum(np.arange(128, 256) * group2) / np.sum(group2)
    var1 = np.sum((np.arange(128) - mean1)**2 * group1) / np.sum(group1)
    var2 = np.sum((np.arange(128, 256) - mean2)**2 * group2) / np.sum(group2)

    likelihood1 = norm.pdf(np.arange(128), mean1, np.sqrt(var1))
    likelihood2 = norm.pdf(np.arange(128, 256), mean2, np.sqrt(var2))

    criterion = np.sum(np.log(likelihood1 * np.sum(group1) / n_pixels + likelihood2 * np.sum(group2) / n_pixels))

    threshold = np.argmax(likelihood1 * np.sum(group1) / n_pixels + likelihood2 * np.sum(group2) / n_pixels)

    return threshold

def niblack_threshold(img, window_size, k, a=10):
    integral_img = np.cumsum(np.cumsum(img, axis=0), axis=1)

    height, width = img.shape[:2]

    up_r = np.maximum(np.arange(height) - window_size // 2, 0)
    dn_r = np.minimum(np.arange(height) + window_size // 2 + 1, height - 1)
    lf_c = np.maximum(np.arange(width) - window_size // 2, 0)
    rt_c = np.minimum(np.arange(width) + window_size // 2 + 1, width - 1)

    sum_ = integral_img[dn_r[:, None], rt_c[None, :]] \
           - integral_img[dn_r[:, None], lf_c[None, :]] \
           - integral_img[up_r[:, None], rt_c[None, :]] \
           + integral_img[up_r[:, None], lf_c[None, :]]
    sum_sq = integral_img[dn_r[:, None], rt_c[None, :]] \
             - integral_img[dn_r[:, None], lf_c[None, :]] \
             - integral_img[up_r[:, None], rt_c[None, :]] \
             + integral_img[up_r[:, None], lf_c[None, :]]

    area = window_size ** 2
    mean = sum_ / area
    var = (sum_sq - mean ** 2) / area

    threshold = mean + k * np.sqrt(var) - a
    mask = img > threshold

    return mask.astype(np.uint8) * 255

def adaptive_niblack_threshold(img, img_label, min_window_size=2, max_window_size=500, metric=cv2.PSNR, k=0.2, a=10):
    best_ql = 0
    best_window = min(img.shape)
    best_bin_img = niblack_threshold(img, best_window, k)

    for window_size in range(min_window_size, max_window_size, 10):
        bin_img = niblack_threshold(img, window_size, k, a)
        qality = metric(bin_img, img_label)
        if best_ql < qality:
            best_bin_img = bin_img
            best_ql = qality
            #best_window = window_size

    return best_bin_img
