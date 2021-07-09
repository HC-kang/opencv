import os 
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/opencv/ch10')
import numpy as np
import cv2

ref = cv2.imread('ref.png', cv2.IMREAD_COLOR)
mask = cv2.imread('mask.bmp', cv2.IMREAD_GRAYSCALE)
ref_ycrcb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

channels = [1,2]
cr_bins = 128
cb_bins = 128
histSize = [cr_binst, cb_bins]
cr_range = [0, 256]
cb_range = [0, 256]
ranges = cr_range + cb_range
hist = cv2.calcHist([ref_ycrcb], channel, mask, histSize, range)

import os 
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/opencv/ch10')
import numpy as np
import cv2


# Calculate CrCb histogram from a reference image

ref = cv2.imread('ref.png', cv2.IMREAD_COLOR)
mask = cv2.imread('mask.bmp', cv2.IMREAD_GRAYSCALE)
ref_ycrcb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

channels = [1, 2]          # Cr, Cb 채널
cr_bins = 128              # 128 등급
cb_bins = 128              # 128 등급
histSize = [cr_bins, cb_bins]      # 128 * 128 사이즈
cr_range = [0, 256]
cb_range = [0, 256]
ranges = cr_range + cb_range

hist = cv2.calcHist([ref_ycrcb], channels, mask, histSize, ranges)  # 기준이 되는 hist 완성

# Apply histogram backprojection to an input image

src = cv2.imread('kids.png', cv2.IMREAD_COLOR)            # hist를 적용할 새 이미지
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)

cv2.imshow('src', src)
cv2.imshow('backproj', backproj)
cv2.waitKey()
cv2.destroyAllWindows()
