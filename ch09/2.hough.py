import os 
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/opencv/ch09')
import numpy as np
import cv2
import math

def hough_lines():
    src = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('사진 불러오기에 실패')
        return

    edge = cv2.Canny(src, 50, 150)
    lines = cv2.HoughLines(edge, 1, math.pi / 180, 250)
    # rho=1, theta=1도, 축열배열의 임계값=250 이상이면 직선으로 판단

    dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x0, y0 = rho * cos_t, rho * sin_t
            alpha = 1000
            pt1 = (int(x0 - alpha * sin_t), int(y0 + alpha * cos_t))
            pt2 = (int(x0 + alpha * sin_t), int(y0 - alpha * cos_t))
            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def hough_line_segments():
    src = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('이미지 불러오기 실패')
        return

    edge = cv2.Canny(src, 50, 150)
    lines = cv2.HoughLinesP(edge, 1, math.pi / 180, 160, minLineLength = 50,
                            maxLineGap=5)
                            # 확률적 허프변환, 최소길이 50, 최대 에지 점 간격은 5

    dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for i in range(lines.shape[0]):
            pt1 = (lines[i][0][0], lines[i][0][1])
            pt2 = (lines[i][0][2], lines[i][0][3])
            cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def hough_circles(): # HoughCircle에서는 sobel과 canny 활용해서 gradient 계산
    src = cv2.imread('coins.png', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('사진 불러오기 실패')
        return
    
    blurred = cv2.blur(src, (3,3))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,param1 = 150, param2=30)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        for i in range(circles.shape[1]):
            cx, cy, radius = circles[0][i]
            cv2.circle(dst, (cx, cy), int(radius), (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hough_lines()
    hough_line_segments()
    hough_circles()


#####
#####
# import numpy as np
# import cv2
# import math


# def hough_lines():
#     src = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

#     if src is None:
#         print('Image load failed!')
#         return

#     edge = cv2.Canny(src, 50, 150)
#     lines = cv2.HoughLines(edge, 1, math.pi / 180, 250)  # rho=1, theta=1도, 축열배열의 임계값=250 이상이면 직선으로 판단

#     dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

#     if lines is not None:
#         for i in range(lines.shape[0]):          # 모든 직선을 두께가 2인 빨간색으로 표시
#             rho = lines[i][0][0]
#             theta = lines[i][0][1]
#             cos_t = math.cos(theta)
#             sin_t = math.sin(theta)
#             x0, y0 = rho * cos_t, rho * sin_t
#             alpha = 1000
#             pt1 = (int(x0 - alpha * sin_t), int(y0 + alpha * cos_t))
#             pt2 = (int(x0 + alpha * sin_t), int(y0 - alpha * cos_t))
#             cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

#     cv2.imshow('src', src)
#     cv2.imshow('dst', dst)
#     cv2.waitKey()
#     cv2.destroyAllWindows()


# def hough_line_segments():
#     src = cv2.imread('building.jpg', cv2.IMREAD_GRAYSCALE)

#     if src is None:
#         print('Image load failed!')
#         return

#     edge = cv2.Canny(src, 50, 150)
#     lines = cv2.HoughLinesP(edge, 1, math.pi / 180, 160, minLineLength=50, maxLineGap=5)  # 확률적 허프변환, 검출할 최소길이는 50, 직선으로 간주할 최대 에지 점 간격은 5

#     dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

#     if lines is not None:
#         for i in range(lines.shape[0]):
#             pt1 = (lines[i][0][0], lines[i][0][1])
#             pt2 = (lines[i][0][2], lines[i][0][3])
#             cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

#     cv2.imshow('src', src)
#     cv2.imshow('dst', dst)
#     cv2.waitKey()
#     cv2.destroyAllWindows()


# def hough_circles():   # cv2.HoughCircles() 내부에서는 Sobel()과 Canny()로 그래디언트와 에지 계산. 
#     src = cv2.imread('coins.png', cv2.IMREAD_GRAYSCALE)

#     if src is None:
#         print('Image load failed!')
#         return

#     blurred = cv2.blur(src, (3, 3))
#     circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,    # 입력 영상과 같은크기(1)의 축열배열, 중심 최소 거리 50 이상일 때만 다른 원으로 인정
#                               param1=150, param2=30)

#     dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

#     if circles is not None:
#         for i in range(circles.shape[1]):
#             cx, cy, radius = circles[0][i]
#             cv2.circle(dst, (cx, cy), int(radius), (0, 0, 255), 2, cv2.LINE_AA)

#     cv2.imshow('src', src)
#     cv2.imshow('dst', dst)
#     cv2.waitKey()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     hough_lines()
#     hough_line_segments()
#     hough_circles()
