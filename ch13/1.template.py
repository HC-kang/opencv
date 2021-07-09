import sys
import numpy as np
import cv2


img = cv2.imread('circuit.bmp', cv2.IMREAD_COLOR)
templ = cv2.imread('crystal.bmp', cv2.IMREAD_COLOR)

if img is None or templ is None:
    print('Image load failed!')
    sys.exit()

img = img + (50, 50, 50)  # 밝기 조절

noise = np.zeros(img.shape, np.int32)
cv2.randn(noise, 0, 10)          # 표준편자가 10인 가우시안 잡음 추가
img = cv2.add(img, noise, dtype=cv2.CV_8UC3)

res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)       # 탬플릿 매칭 수행
res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # 매칭 결과를 0~255 사이로 정규화, 타입을 CV_8U로 변환

_, maxv, _, maxloc = cv2.minMaxLoc(res)   # res 행렬의 최대값 위치를 찾아 maxloc에 저장
print('maxv:', maxv)

(th, tw) = templ.shape[:2]
cv2.rectangle(img, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 0, 255), 2)  # 매칭 결과를 빨간색 사각형으로 표시

cv2.imshow('templ', templ)
cv2.imshow('res_norm', res_norm)
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
