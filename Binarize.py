import numpy as np
from scipy.stats import norm
import cv2
from Metrics import *


class BinarizationMethod():
    NIBLACK = 'Niblack'
    OTSU = 'Otsu'
    OTSU_L = 'Otsu_l'
    NIBLACK_MULTISCALE = 'Niblack_multiscale'




def otsu_thresholding(image):
    hist, bins = np.histogram(image, bins=256, range=(0, 255))
    n_pixels = np.sum(hist)

    norm_hist = hist / n_pixels

    # mean = np.mean(img)
    max_var = 0
    threshold = 0
    for i in range(256):

        w0 = np.sum(norm_hist[:i])
        w1 = np.sum(norm_hist[i:])

        u0 = np.sum(norm_hist[:i] * np.arange(i)) / w0 if w0 > 0 else 0
        u1 = np.sum(norm_hist[i:] * np.arange(i, 256)) / w1 if w1 > 0 else 0

        var = w0 * w1 * (u0 - u1) ** 2

        if var > max_var:
            max_var = var
            threshold = i
    mask = np.where(image > threshold, 1, 0)

    return mask.astype(np.uint8) * 255


def maximum_likelihood_thresholding(image, threshold=128):
    hist, bins = np.histogram(image, bins=256, range=(0, 255))
    n_pixels = np.sum(hist)
    group1 = hist[:threshold]
    group2 = hist[threshold:]

    mean1 = np.sum(np.arange(threshold) * group1) / np.sum(group1)
    mean2 = np.sum(np.arange(threshold, 256) * group2) / np.sum(group2)
    var1 = np.sum((np.arange(threshold) - mean1) ** 2 * group1) / np.sum(group1)
    var2 = np.sum((np.arange(threshold, 256) - mean2) ** 2 * group2) / np.sum(group2)

    likelihood1 = norm.pdf(np.arange(threshold), mean1, np.sqrt(var1))
    likelihood2 = norm.pdf(np.arange(threshold, 256), mean2, np.sqrt(var2))

    threshold = np.argmax(likelihood1 * np.sum(group1) / n_pixels + likelihood2 * np.sum(group2) / n_pixels)
    mask = image > threshold

    return mask.astype(np.uint8) * 255


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


def adaptive_niblack_threshold(img, min_window_size=10, max_window_size=500, k=0.2, a=10, scale_factor= 0.5):
    thresholds = []
    for window_size in range(min_window_size, max_window_size, 2):
        threshold = niblack_threshold(img, window_size, k, a)
        thresholds.append(threshold)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    best_thresholds = np.zeros(img.shape, dtype=np.uint8)
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        scaled_threshold = cv2.resize(threshold, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        best_thresholds = np.maximum(best_thresholds, scaled_threshold)

    return best_thresholds
def binarize(image, method, window_size, min_window_size, max_window_size, k, a, scale_factor):
    if method == BinarizationMethod.NIBLACK:
        bin_image = niblack_threshold(image, window_size, k, a)
    elif method == BinarizationMethod.OTSU:
        bin_image = otsu_thresholding(image)
    elif method == BinarizationMethod.OTSU_L:
        bin_image = maximum_likelihood_thresholding(image)
    elif method == BinarizationMethod.NIBLACK_MULTISCALE:
        bin_image = adaptive_niblack_threshold(image, min_window_size, max_window_size, k, a, scale_factor)
    else:
        raise ValueError('Unknown binarization method')
    return bin_image

