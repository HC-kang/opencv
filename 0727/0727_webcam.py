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
        frame = pencil_filter(frame)

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

def cartoon_filter(img):
    blr = cv2.bilateralFilter(img, -1, 20, 7)
    edge = 255-cv2.Canny(blr, 50, 150)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    dst = cv2.bitwise_and(blr, edge)
    return dst

def pencil_filter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (0, 0), 3)
    dst = cv2.divide(gray, blr, scale = 255)
    return dst

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
        frame = pencil_filter(frame)

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

###################
import cv2
import os
import numpy as np

os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/cam_DL_opencv')

filename = '/Users/heechankang/projects/pythonworkspace/git_study/cam_DL_opencv/cat.bmp'

img = cv2.imread(filename)

if img is None:
    print('이미지 불러오기 실패')
    sys.exit()

model = './googlenet/bvlc_googlenet.caffemodel'
config = './googlenet/deploy.prototxt'

net = cv2.dnn.readNet(model, config)

classNames = []

with open('googlenet/classification_classes_ILSVRC2012.txt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)

blob = cv2.dnn.blobFromImage(img, 1, (244, 244), (104, 117, 123), swapRB = False)
net.setInput(blob)
prob = net.forward()

out = prob.flatten()
# print(out)
classId = np.argmax(out)
confidence = out[classId]

text = f'{classNames[classId]} ({confidence * 100:4.2f}%)'
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()


#####################################

import os 
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/cam_DL_opencv/face_detector')
import numpy as np
import cv2

model = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
config = 'deploy.prototxt'


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    net.setInput(blob)
    out = net.forward()# out.shape=(1,1, 200, 7)
    
    
    detect = out[0, 0, :, :] ##0, 0, 사용안함
    (h, w) = frame.shape[:2]

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]
        if confidence < 0.5:
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = f'Face: {confidence:4.2f}'
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

        
cap.release()
cv2.destroyAllWindows()

########################
import sys
import numpy as np
import cv2
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/cam_DL_opencv')


# 모델 & 설정 파일
model = 'yolo_v3/yolov3.weights'
config = 'yolo_v3/yolov3.cfg'
class_labels = 'yolo_v3/coco.names'

confThreshold = 0.5
nmsThreshold = 0.4

# 테스트 이미지 파일
img_files = ['yolo_v3/dog.jpg', 'yolo_v3/person.jpg', 
             'yolo_v3/sheep.jpg', 'yolo_v3/kite.jpg']

# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# 클래스 이름 불러오기

classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 출력 레이어 이름 받아오기

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# output_layers = ['yolo_82', 'yolo_94', 'yolo_106']

# 실행

for f in img_files:
    img = cv2.imread(f)

    if img is None:
        continue

    # 블롭 생성 & 추론
    blob = cv2.dnn.blobFromImage(img, 1/255., (320, 320), swapRB=True)
    net.setInput(blob)
    outs = net.forward(output_layers) #

    # outs는 3개의 ndarray 리스트.
    # outs[0].shape=(507, 85), 13*13*3=507
    # outs[1].shape=(2028, 85), 26*26*3=2028
    # outs[2].shape=(8112, 85), 52*52*3=8112

    h, w = img.shape[:2]

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            # detection: 4(bounding box) + 1(objectness_score) + 80(class confidence)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                # 바운딩 박스 중심 좌표 & 박스 크기
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)

                # 바운딩 박스 좌상단 좌표
                sx = int(cx - bw / 2)
                sy = int(cy - bh / 2)

                boxes.append([sx, sy, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # 비최대 억제, Non Max Suppression
#     https://www.visiongeek.io/2018/07/yolo-object-detection-opencv-python.html
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        sx, sy, bw, bh = boxes[i]
        label = f'{classes[class_ids[i]]}: {confidences[i]:.2}'
        color = colors[class_ids[i]]
        cv2.rectangle(img, (sx, sy, bw, bh), color, 2)
        cv2.putText(img, label, (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()


##################
import sys
import numpy as np
import cv2
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/cam_DL_opencv')



# 비디오 파일 열기
cap = cv2.VideoCapture('PETS2000.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 배경 영상 등록
ret, back = cap.read()

if not ret:
    print('Background image registration failed!')
    sys.exit()

back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0, 0), 1.0)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # 차영상 구하기 & 이진화
    diff = cv2.absdiff(gray, back)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()

############
import sys
import numpy as np
import cv2
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/cam_DL_opencv')



# 비디오 파일 열기
cap = cv2.VideoCapture('PETS2000.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 배경 영상 등록
ret, back = cap.read()

if not ret:
    print('Background image registration failed!')
    sys.exit()

back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0, 0), 1.0)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # 차영상 구하기 & 이진화
    diff = cv2.absdiff(gray, back)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 레이블링을 이용하여 바운딩 박스 표시
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)

    for i in range(1, cnt):
        x, y, w, h, s = stats[i]

        if s < 100:
            continue

        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()

########
import sys
import numpy as np
import cv2
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/cam_DL_opencv')



# 비디오 파일 열기
cap = cv2.VideoCapture('PETS2000.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 배경 영상 등록
ret, back = cap.read()

if not ret:
    print('Background image registration failed!')
    sys.exit()

# back: uint8 배경, fback: float32 배경
back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0, 0), 1.0)
fback = back.astype(np.float32)

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    # fback: float32, back: uint8 배경
    fback2 = cv2.accumulateWeighted(gray, fback, 0.01)
    back = fback2.astype(np.uint8)

    diff = cv2.absdiff(gray, back)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 레이블링을 이용하여 바운딩 박스 표시
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)

    for i in range(1, cnt):
        x, y, w, h, s = stats[i]

        if s < 100:
            continue

        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)
    cv2.imshow('back', back)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()

#################
import sys
import numpy as np
import cv2
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/cam_DL_opencv')



src1 = cv2.imread('frame1.jpg')
src2 = cv2.imread('frame2.jpg')

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

gray1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)

# https://bskyvision.com/668
pt1 = cv2.goodFeaturesToTrack(gray1, 50, 0.01, 10) #코너점을 찾는 함수
# print(pt1)

pt2, status, err = cv2.calcOpticalFlowPyrLK(src1, src2, pt1, None) #출력

print(pt2.shape)
# center_1 = pt1[1, 0].astype(np.uint32)
# # center_1 = center_1.astype(np.uint8)
# print(pt1[1, 0], center_1)


# addWeighted(src1, alpha, src2, beta, gamma) -> dst
dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)


for i in range(pt2.shape[0]):
    if status[i, 0] == 0:
        continue

        #version 4.5.2
    center_1 = pt1[i, 0].astype(np.uint32)
    center_2 = pt2[i, 0].astype(np.uint32)
    
#     circle(img, center, radius, color, thickness, lineType)
    cv2.circle(dst, tuple(center_1), 4, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(dst, tuple(center_2), 4, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.arrowedLine(dst, tuple(center_1), tuple(center_2), (0, 255, 0), 2)

    
    
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

#################
import sys
import numpy as np
import cv2


# 카메라 장치 열기
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

# 설정 변수 정의
MAX_COUNT = 50
needToInit = False
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 128)]

ptSrc = None
ptDst = None

# 카메라 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if needToInit:
        ptSrc = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
        needToInit = False
        
        
    if ptSrc is not None: # not Null
        if prev is None:
            prev = gray.copy()

        ptDst, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, ptSrc, None)

        for i in range(ptDst.shape[0]):
            if status[i, 0] == 0:
                continue

            center = ptDst[i, 0].astype(np.uint32)
            cv2.circle(img, tuple(center), 4, colors[i % 8], 2, cv2.LINE_AA)

    cv2.imshow('frame', img)

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord('r'):
        needToInit = not needToInit
        

    ptDst, ptSrc = ptSrc, ptDst
    prev = gray


cap.release()
cv2.destroyAllWindows()
