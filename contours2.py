import cv2
import os, glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import function as fc

#画像ファイルの取得
classlabel = 'sozu'
photos_dir = './' + classlabel
files = []
files = glob.glob(photos_dir + '/*.jpg')

# 画像を読み込む。表示する。
fig = plt.figure()
img = cv2.imread(str(files[0]))
# plt.imshow(img)
#plt.show()

ax1 = fig.add_subplot(231)
ax1.imshow(img)
ax1.axis('off')

#ヒストグラム表示
tailes_gray = fc.to_grayscale(str(files[0]))
hist = cv2.calcHist([tailes_gray],[0],None,[256],[0,256])
ax2 = fig.add_subplot(232)
ax2.plot(hist)
ax2.set_xlim([0,256])


#二値化
th_tailes = fc.binary_threshold(str(files[0]), 150)
ax3 = fig.add_subplot(233)
ax3.imshow(th_tailes, 'gray')
ax3.axis('off')

# th_tailes2 = fc.binary_threshold2(str(files[0]))
# mo_tailes2 = fc.morph(th_tailes)
# mo_tailes2 = cv2.bitwise_not(mo_tailes2)

# 輪郭を抽出する。
# mode引数の変更
# contours, hierarchy = cv2.findContours(th_tailes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(th_tailes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(th_tailes, cv2.cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(th_tailes, cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(contours)
print(len(contours))

# 輪郭表示する。
ax4 = fig.add_subplot(234)
fc.draw_contours(ax4, img, contours)

ax5 = fig.add_subplot(235)
cnt = contours[0]
# 輪郭に外接する長方形を取得する。
x,y,w,h = cv2.boundingRect(cnt)
# 長方形を描画する。
img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
ax5.imshow(img)

#輪郭部切り出し
cutting_img = img[y:y+h,x:x+w]
# ax6 = fig.add_plot(
plt.imshow(cutting_img)

width = cutting_img.shape[0]
height = cutting_img.shape[1]
resize_img = cv2.resize(cutting_img,(height*2,width*2))
plt.imshow(resize_img)
plt.show()
#ax5 = fig.add_subplot(235)
#ax5.imshow(resize_img)
#ax5.axis('off')

#並列表示
plt.show(fig)

# 各 method での輪郭抽出の結果を描画する。
# methods = {'cv2.CHAIN_APPROX_NONE': cv2.CHAIN_APPROX_NONE,
#           'cv2.CHAIN_APPROX_SIMPLE': cv2.CHAIN_APPROX_SIMPLE,
#           'cv2.CHAIN_APPROX_TC89_L1': cv2.CHAIN_APPROX_TC89_L1,
#           'cv2.CHAIN_APPROX_TC89_KCOS': cv2.CHAIN_APPROX_TC89_KCOS}

# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# for ax, (name, method) in zip(axes.ravel(), methods.items()):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    contours, hierarchy = cv2.findContours(mo_tailes2, cv2.RETR_EXTERNAL, method)
#    ax.set_title(name)
#    fc.draw_contours(ax, img, contours)
# plt.show()
