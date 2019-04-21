import cv2
import os, glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


#BGR→グレースケールへ変更
def to_grayscale(path):
    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed

# 二値化
def binary_threshold(path, threshold):
    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,th = cv2.threshold(grayed,threshold,255,cv2.THRESH_BINARY)
    # th = cv2.adaptiveThreshold(grayed,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return th

#　適用閾値、周辺地域の平均値
def binary_threshold2(path):
    img = cv2.imread(path)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img,5)
    th = cv2.adaptiveThreshold(grayed,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    return th

# 平滑化
def morph(img):
    kernel = np.ones((2, 1),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations=1)
    # closing = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel,iterations=1)
    return dilation



# 輪郭を元画像に表示する
def draw_contours(ax, img, contours):
    ax.imshow(img)
    ax.axis('off')
    for i, cnt in enumerate(contours):
        cnt = np.squeeze(cnt, axis=1)  # (NumPoints, 1, 2) -> (NumPoints, 2)
        # 輪郭の点同士を結ぶ線を描画する。
        ax.add_patch(Polygon(cnt, color='b', fill=None, lw=2))
        # 輪郭の点を描画する。
        # ax.plot(cnt[:, 0], cnt[:, 1], 'ro', mew=0, ms=4)
        # 輪郭の番号を描画する。
        # ax.text(cnt[0][0], cnt[0][1], i, color='orange', size='20')
