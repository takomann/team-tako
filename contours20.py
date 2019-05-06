import cv2
import os, glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
contours, hierarchy = cv2.findContours(th_tailes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(th_tailes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(th_tailes, cv2.cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(th_tailes, cv2.cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭表示する。
ax4 = fig.add_subplot(234)
fc.draw_contours(ax4, img, contours)

#並列表示
plt.show(fig)

#ax5 = fig.add_subplot(235)

X=[]
Y=[]
W=[]
H=[]
tailes_photo =[]
area = []


for i, cnt in enumerate(contours):
    # 輪郭に外接する長方形の情報を取得する。
    x,y,w,h = cv2.boundingRect(cnt)
    X.append(x)
    Y.append(y)
    W.append(w)
    H.append(h)
#    ax4.add_patch(patches.Rectangle(xy=(x,y),width=w,height=h,color='g',fill=None,lw=2))
    # 長方形を描画する。
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cutting_img = img[y:y+h,x:x+w]
    #切り取り画像の情報を格納

    tailes_photo.append(cutting_img)
    plt.imshow(cutting_img)
    plt.show()

photo0 = tailes_photo[0]
photo1 = tailes_photo[1]
#print(photo0.shape)
#print(photo1.shape)
#print(tailes_photo[0])
#print(len(tailes_photo[1]))
# 画像の分割を検討　
size = photo1.shape[0]
v_size = photo0.shape[0]
h_size = photo0.shape[1] // size * size
photo0 = photo0[:v_size, :h_size]
h_split = photo0.shape[1] // size
out_img = []
out_img.extend(np.hsplit(photo0, h_split))

for i, outimg in enumerate(out_img):
    plt.imshow(outimg)
    plt.show()


print(X)
print(Y)
print(W)
print(H)
#print(len(tailes_photo))
#plt.hist(area,bins=10)
