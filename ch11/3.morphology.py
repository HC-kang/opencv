import os 
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/opencv/ch11')
import numpy as np
import cv2
import matplotlib.pyplot as plt


def erode_dilate():
    src = cv2.imread('milkdrop.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('이미지 불러오기 실패')
        return
    
    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dst1 = cv2.erode(src_bin, None)
    dst2 = cv2.dilate(src_bin, None)

    plt.subplot(221), plt.axis('off'), plt.imshow(src, 'gray'), plt.title('src')
    plt.subplot(222), plt.axis('off'), plt.imshow(src_bin, 'gray'), plt.title('src_bin')
    plt.subplot(223), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('erode')
    plt.subplot(224), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('dilate')
    plt.show()
    

def open_close():
    src = cv2.imread('milkdrop.bmp', cv2.IMREAD_GRAYSCALE)

    if src is None:
        print('이미지 불러오기 실패')
        return

    _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dst1 = cv2.morphologyEx(src_bin, cv2.MORPH_OPEN, None)
    dst2 = cv2.morphologyEx(src_bin, cv2.MORPH_CLOSE, None)

    plt.subplot(221), plt.axis('off'), plt.imshow(src, 'gray'), plt.title('src')
    plt.subplot(222), plt.axis('off'), plt.imshow(src_bin, 'gray'), plt.title('src_bin')
    plt.subplot(223), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('open')
    plt.subplot(224), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('close')
    plt.show()


if __name__ == '__main__':
    erode_dilate()
    open_close()



#####
#####
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt


# def erode_dilate():
#     src = cv2.imread('milkdrop.bmp', cv2.IMREAD_GRAYSCALE)

#     if src is None:
#         print('Image load failed!')
#         return

#     _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     dst1 = cv2.erode(src_bin, None)          # src_bin에 대해 침식과 팽창 수행
#     dst2 = cv2.dilate(src_bin, None)

#     plt.subplot(221), plt.axis('off'), plt.imshow(src, 'gray'), plt.title('src')
#     plt.subplot(222), plt.axis('off'), plt.imshow(src_bin, 'gray'), plt.title('src_bin')    # 단순 이진화
#     plt.subplot(223), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('erode')         # 침식
#     plt.subplot(224), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('dilate')        # 팽창
#     plt.show()


# def open_close():
#     src = cv2.imread('milkdrop.bmp', cv2.IMREAD_GRAYSCALE)

#     if src is None:
#         print('Image load failed!')
#         return

#     _, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     dst1 = cv2.morphologyEx(src_bin, cv2.MORPH_OPEN, None)
#     dst2 = cv2.morphologyEx(src_bin, cv2.MORPH_CLOSE, None)


#     plt.subplot(221), plt.axis('off'), plt.imshow(src, 'gray'), plt.title('src')
#     plt.subplot(222), plt.axis('off'), plt.imshow(src_bin, 'gray'), plt.title('src_bin')
#     plt.subplot(223), plt.axis('off'), plt.imshow(dst1, 'gray'), plt.title('open')        # 열기
#     plt.subplot(224), plt.axis('off'), plt.imshow(dst2, 'gray'), plt.title('close')       # 닫기
#     plt.show()

# if __name__ == '__main__':
#     erode_dilate()
#     open_close()
