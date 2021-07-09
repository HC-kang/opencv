import sys
import cv2
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/opencv/ch02')

print('Hello OpenCV', cv2.__version__)

img = cv2.imread('lenna.bmp')

if img is None:
    print('Image load failed! ')
    sys.exit()

cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()

import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()               # 이유?

