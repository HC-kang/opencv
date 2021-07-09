import numpy as np
import cv2
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/opencv/ch03')



def func1():
    img1 = cv2.imread('cat.bmp', cv2.IMREAD_GRAYSCALE)       # 컬러를 흑백으로

    if img1 is None:
        print('Image load failed!')
        return

    print('type(img1):', type(img1))                    # <class 'numpy.ndarray'>
    print('img1.shape:', img1.shape)                 # (480, 640)

    if len(img1.shape) == 2:
        print('img1 is a grayscale image')
    elif len(img1.shape) == 3:
        print('img1 is a truecolor image')

    cv2.imshow('img1', img1)                            # gray image 출력
    cv2.waitKey()
    cv2.destroyAllWindows()


def func2():
    img1 = np.empty((480, 640), np.uint8)       # grayscale image
    img2 = np.zeros((480, 640, 3), np.uint8)    # color image
    img3 = np.ones((480, 640), np.int32)        # 1's matrix
    img4 = np.full((480, 640), 0, np.float32)   # Fill with 0.0

    mat1 = np.array([[11, 12, 13, 14],
                     [21, 22, 23, 24],
                     [31, 32, 33, 34]]).astype(np.uint8)

    mat1[0, 1] = 100    # element at x=1, y=0
    mat1[2, :] = 200

    print(mat1)
    cv2.imshow('mat1', mat1)
    cv2.waitKey()
    cv2.destroyAllWindows()


def func3():
    img1 = cv2.imread('cat.bmp')

    img2 = img1             # img2는 img1을 직접 가리킴
    img3 = img1.copy()      # img3는 복사본
    print(img1.shape)

    img1[:, :] = (0, 255, 255)  # yellow, broadcast
    #img1[:, :, :] = (0, 255, 255)  # yellow, broadcast. 같은 결과

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    cv2.waitKey()
    cv2.destroyAllWindows()


def func4():
    img1 = cv2.imread('lenna.bmp', cv2.IMREAD_GRAYSCALE)

    img2 = img1[200:400, 200:400]           #slicing
    img3 = img1[200:400, 200:400].copy()
    img4 = 255-img2                         # 영상반전 (C코드는 ~img2)

    img2 += 20                  # 좀 더 밝게

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)    # 밝아진 이미지
    cv2.imshow('img3', img3)
    cv2.imshow('img4', img4)
    cv2.waitKey()
    cv2.destroyAllWindows()


def func5():
    mat1 = np.array(np.arange(12)).reshape(3, 4)

    print('mat1:')
    print(mat1)

    h, w = mat1.shape[:2]

    mat2 = np.zeros(mat1.shape, type(mat1)).astype(np.uint8)

    for j in range(h):
        for i in range(w):
            mat2[j, i] = mat1[j, i] * 20    # 밝기조절

    print('mat2:')
    print(mat2)
    cv2.imshow('mat2', mat2)
    cv2.waitKey()
    cv2.destroyAllWindows()

def func6():
    mat1 = np.ones((3, 4), np.int32)    # 1's matrix
    mat2 = np.arange(12).reshape(3, 4)
    mat3 = mat1 + mat2
    mat4 = (mat2 * 20).astype(np.uint8)    # 출력을 위해서

    print("mat1:", mat1, sep='\n')
    print("mat2:", mat2, sep='\n')
    print("mat3:", mat3, sep='\n')
    print("mat4:", mat4, sep='\n')
    cv2.imshow('mat4', mat4)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    func1()
    func2()
    func3()
    func4()
    func5()
    func6()
