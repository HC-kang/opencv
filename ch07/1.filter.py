import os 
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/opencv/ch07')
import sys
import numpy as np
import cv2

src = cv2.imread('rose.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('사진 불러오기 실패')
    sys.exit()

emboss = np.array([[-1, -1, 0],
                   [-1, 0, 1],
                   [0, 1, 1]], np.float32)

dst = cv2.filter2D(src, -1, emboss, delta = 128) # 128 더해서 중간값으로 만들어주기.

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
    


#####
#####
# import sys
# import numpy as np
# import cv2


# src = cv2.imread('rose.bmp', cv2.IMREAD_GRAYSCALE)

# if src is None:
#     print('Image load failed!')
#     sys.exit()

# emboss = np.array([[-1, -1, 0],
#                    [-1, 0, 1],
#                    [0, 1, 1]], np.float32)

# dst = cv2.filter2D(src, -1, emboss, delta=128)

# cv2.imshow('src', src)
# cv2.imshow('dst', dst)

# cv2.waitKey()
# cv2.destroyAllWindows()
