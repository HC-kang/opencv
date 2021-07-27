import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

from numpy.lib.twodim_base import _vander_dispatcher
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/opencv/0727/cam_DL_opencv')

print(cv2.__version__)

# img = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('cat.bmp', cv2.IMREAD_COLOR)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow('image')
cv2.imshow('image', img)

cv2.imwrite('cat_test.jpg', img)

while True:
    if cv2.waitKey() == 27:
        break
    # if cv2.waitKey() == ord('q'):
    #     break
cv2.destroyAllWindows()


imgGray = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)

imgBGR = cv2.imread('cat.bmp', cv2.IMREAD_COLOR)
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

plt.subplot(131)
plt.imshow(imgGray, cmap = 'gray')
plt.axis('off')
plt.subplot(132)
plt.imshow(imgBGR)
plt.axis('off')
plt.subplot(133)
plt.imshow(imgRGB)
plt.axis('off')


import glob
img_files = glob.glob('./images/*.jpg')

if img_files is None:
    print('images load failed')
    sys.exit()

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

index = 0
while True:
    img = cv2.imread(img_files[index], cv2.IMREAD_COLOR)

    cv2.imshow('image', img)

    if cv2.waitKey(3000) == 27:
        break

    index+=1
    if index>=len(img_files):
        index = 0
        break

cv2.destroyAllWindows()

#########################


import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

# img = cv2.imread('cat.bmp')
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print('camera open failed')
    sys.exit()

# cv2.namedWindow('image')
# cv2.namedWindow('image_edge')
cv2.namedWindow('image_inverse')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # edge = cv2.Canny(frame, 50, 150)
    inversed = ~frame

    # cv2.imshow('iamge', frame)
    # cv2.imshow('iamge_edge', edge)
    cv2.imshow('iamge_inverse', inversed)

    if cv2.waitKey(10) == 27:
        break

# cv2.imshow('image', img)
# cv2.waitKey()
cap.release()
cv2.destroyAllWindows()


#######################

import numpy as np
import sys, cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print('camera open failed')
    sys.exit()

cv2.namedWindow('image')
cv2.namedWindow('image_edge')

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = 30

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h), isColor = False)
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h), isColor = True)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print('image read failed')
        break

    edge = cv2.Canny(frame, 50, 150)
    edge_color = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    cv2.imshow('image', frame)
    # cv2.imshow('image_edge', edge)
    cv2.imshow('image_edge', edge_color)

    # out.write(edge)
    out.write(edge_color)

    if cv2.waitKey(10) == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()

####################


import sys 
import numpy as np
import cv2


def cartoon_filter(img):
    h, w = img.shape[:2]
#     img2 = cv2.resize(img, (w//2, h//2))

#     https://ailearningcentre.wordpress.com/2017/05/07/bilateral-filter/
    blr = cv2.bilateralFilter(img, -1, 20, 7)
    edge = 255 - cv2.Canny(img, 80, 120)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

#     https://copycoding.tistory.com/156
    dst = cv2.bitwise_and(blr, edge)
#     dst = cv2.resize(dst, (w, h), interpolation=cv2.INTER_NEAREST)

    return dst


def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), 3)
    dst = cv2.divide(gray, blr, scale=255)
    return dst


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print('video open failed!')
    sys.exit()

cam_mode = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if cam_mode == 1:
        frame = cartoon_filter(frame)
    elif cam_mode == 2:
        frame = pencil_sketch(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord(' '):
        cam_mode += 1
        if cam_mode == 3:
            cam_mode = 0


cap.release()
cv2.destroyAllWindows()

##############
import numpy as np
import sys
import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print('camera open failed')
    sys.exit()

cv2.namedWindow('image')

cam_mode = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if cam_mode == 1:
        frame = cartoon_filter(frame)
    elif cam_mode ==2:
        frame = pencil_sketch(frame)

    cv2.imshow('image', frame)

    key = cv2.waitKey(10)

    if key == 27:
        break
    elif key == ord(' '):
        cam_mode += 1
        if cam_mode >= 3:
            cam_mode = 0

cap.release()
# out.release()
cv2.destroyAllWindows()

#########

import numpy as np
import sys
import cv2

def cartoon_filter(frame):
    blr = cv2.bilateralFilter(src, )