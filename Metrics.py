from enum import Enum
import cv2
import numpy as np
from math import sqrt





def Creat_W(n):
    m = 2 * n + 1
    i_c = (m - 1) // 2
    j_c = (m - 1) // 2
    W = np.array(
        [[1 / sqrt((i - i_c) ** 2 + (j - j_c) ** 2) if (abs(j - j_c) + abs(i - i_c)) != 0 else 0 for i in range(m)] for
         j in range(m)])
    W[i_c][j_c] = 0
    W_sum = np.sum(W)
    W = W / W_sum
    return W


def DRDblock(i, j, W, m, g, f):
    xf = f.shape[0]
    yf = f.shape[1]
    value_gk = g[i][j]
    Bk = np.zeros((m, m))
    Dk = np.zeros((m, m))
    h = m
    for x in range(m - 1):
        for y in range(m - 1):
            if (i - h + x < 1 or j - h + y < 1 or i - h + x > xf or j - h + y > yf):
                Bk[x][y] = value_gk
            else:
                Bk[x][y] = f[i - h + x - 1][j - h + y - 1];
            Dk[x][y] = abs(Bk[x][y] - value_gk)
    DRDk = np.sum(Dk * W)
    return DRDk


def nubm_calc(f, ii, jj, blck):
    startx = (ii - 1) * blck + 1
    endx = ii * blck
    starty = (jj - 1) * blck + 1
    endy = jj * blck
    check_prv = -2
    retb = 0
    for xx in range(startx, endx - 1):
        for yy in range(startx, endx - 1):
            check = f[xx - 1][yy - 1]
            if (check_prv < 0):
                check_prv = check
            elif (check != check_prv):
                retb = 1
                break
        if (retb != 0):
            break
    return retb


def DRD( pred,gt, n=2, SIZE_BLOCK=8):
    W = Creat_W(n)
    if np.max(pred) != 1:
        pred[pred < 127] = 0
        pred[pred > 127] = 1
    if np.max(gt) != 1:
        gt[gt < 127] = 0
        gt[gt > 127] = 1

    # assert(set(gt)==2, set(pred)==2)
    error_map = np.abs(gt.astype('int32') - pred.astype('int32'))  # //255
    error_padded = np.pad(error_map, ((n, n), (n, n)), mode='constant', constant_values=0)
    DRDk_sum = 0
    NUBM = 0

    for i in range(n, error_map.shape[0] + n):
        for j in range(n, error_map.shape[1] + n):
            if error_padded[i][j] == 1:
                DRDk_sum += np.sum(error_padded[i - n:i + n + 1, j - n:j + n + 1] * W)

    for i in range(0, gt.shape[0] // SIZE_BLOCK * SIZE_BLOCK, SIZE_BLOCK):

        for j in range(0, gt.shape[1] // SIZE_BLOCK * SIZE_BLOCK, SIZE_BLOCK):

            if np.sum(gt[i:i + SIZE_BLOCK, j:j + SIZE_BLOCK]) != SIZE_BLOCK * SIZE_BLOCK and np.sum(
                    gt[i:i + SIZE_BLOCK, j:j + SIZE_BLOCK]) != 0:
                NUBM += 1

    if NUBM == 0:
        NUBM = 1
    return DRDk_sum/NUBM


def PSNR(image, label):

    mse = np.mean((image.astype(np.float64) / 255 - label.astype(np.float64) / 255) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10(1. / mse)

def ignore_area(image, label):
     label[label == 128] = image[label == 128]
     return label
def Metrics(metric_name, image, label):
    label = ignore_area(image, label)

    if metric_name == 'PSNR':
        metric = PSNR(image, label)
    elif metric_name == 'DRD':
        metric = DRD(image, label)
    else:
        metric = None
    return  metric
