import cv2
import os, glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Polygon, Rectangle
import function as fc

#画像ファイルの取得
classlabel = 'sozu'
photos_dir = './' + classlabel
files = []
files = glob.glob(photos_dir + '/*.jpg')

# 画像を読み込む。表示する。
fig = plt.figure()
img = cv2.imread(str(files[0]))
ax1 = fig.add_subplot(231)
ax1.imshow(img)
ax1.axis('off')

#ヒストグラム表示
##tailes_gray = fc.to_grayscale(str(files[0]))
##hist = cv2.calcHist([tailes_gray],[0],None,[256],[0,256])
##ax2 = fig.add_subplot(232)
##ax2.plot(hist)
##ax2.set_xlim([0,256])

#二値化
th_tailes = fc.binary_threshold(str(files[0]), 150)
ax3 = fig.add_subplot(233)
ax3.imshow(th_tailes, 'gray')
ax3.axis('off')

# 輪郭を抽出する。
# mode引数の変更
contours, hierarchy = cv2.findContours(th_tailes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(th_tailes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(th_tailes, cv2.cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(th_tailes, cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭表示する。
ax4 = fig.add_subplot(234)
fc.draw_contours(ax4, img, contours)

#並列表示
plt.show(fig)

X=[]
Y=[]
W=[]
H=[]
tailes_photo =[]
area = []

for i, cnt in enumerate(contours):
    # 輪郭に外接する回転した長方形を取得する。
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (width, height), angle = rect
    print('bounding box of contour {} =>''center: ({:.2f}, {:.2f}), size: ({:.2f}, {:.2f}), angle: {:.2f}'.format(i, cx, cy, width, height, angle))
    # 回転した長方形の4点の座標を取得する。
    rect_points = cv2.boxPoints(rect)
    # 回転した長方形を描画する。
    print(rect_points)
    ax4.add_patch(Polygon(rect_points, color='g', fill=None, lw=2))
    #牌の部分の切り出し
    cx = int(cx)
    cy = int(cy)
    width = int(width)
    height = int(height)
    angle = int(angle)
    cutting_img = fc.cut_after_rot(img, (angle), (cx, cy), (width, height))
    tailes_photo.append(cutting_img)
    plt.imshow(cutting_img)
    plt.show()


#　緑一色Ver.　横一列の牌の切り離し(自摸を分離しないとき)
if len(tailes_photo) == 1:
    photo0 = tailes_photo[0]
    # 画像の分割
    taile = []
    X = photo0.shape[1]
    x = 0
    tailes_number = 14
    width = photo0.shape[1] // tailes_number
    for i in range(0,tailes_number):
        if i <= 7:
            tailes = photo0[0:photo0.shape[0],x:x+width]
            x = x + width
            #切り取り画像の情報を格納
            taile.append(tailes)
        else:
            tailes = photo0[0:photo0.shape[0],X-width:X]
            X = X - width
            #切り取り画像の情報を格納
            taile.append(tailes)

    for i, tl in enumerate(taile):
        photo_path = photos_dir + '/' + str(i) + '.jpg'
        cv2.imwrite(photo_path,tl)

# タンヤオVer.（自摸を分離するとき）
if len(tailes_photo) == 2:
    photo0 = tailes_photo[0]
    photo1 = tailes_photo[1]
    print(photo0.shape)
    print(photo1.shape)



    # 画像の分割
    taile = [photo1]
    X = photo0.shape[1]
    x = 0
    tailes_number = 13
    width = photo1.shape[1]
    for i in range(0,tailes_number):
        if i <= 6:
            tailes = photo0[0:photo0.shape[0],x:x+width]
            x = x + width
            #切り取り画像の情報を格納
            taile.append(tailes)
        else:
            tailes = photo0[0:photo0.shape[0],X-width:X]
            X = X - width
            #切り取り画像の情報を格納
            taile.append(tailes)

    for i, tl in enumerate(taile):
        photo_path = photos_dir + '/' + str(i) + '.jpg'
        cv2.imwrite(photo_path,tl)
