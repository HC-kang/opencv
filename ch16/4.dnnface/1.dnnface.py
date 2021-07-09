import sys
import numpy as np
import cv2

Lib = 'caffe'   # 'caffe' or 'tensorflow'
if Lib == 'caffe':
    model = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config = 'deploy.prototxt'
else:
    model = 'opencv_face_detector_uint8.pb'
    config = 'opencv_face_detector.pbtxt'

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

while True:
    _, frame = cap.read()
    if frame is None:
        break

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123))
    # SSD 기본 네트워크 구조에서 입력 영상 크기는 300×300, 영상 평균값으로 Scalar(104, 177, 123)을 사용
    # blob 블롭 객체는 1×3×300×300 형태의 4차원 행렬이며, net 객체에 입력으로 사용됨

    net.setInput(blob)
    detect = net.forward()   # detect 행렬은 1×1×N×7 크기의 4차원 행렬 (1, 1, 7은 고정값이고, N은 얼굴 개수로 최대 200개)

    (h, w) = frame.shape[:2]
    detect = detect[0, 0, :, :]     # detect 행렬 1×1×N×7에서 N×7 부분만 이용하면 됨. N개 얼굴은 신뢰도가 높은 값부터 정렬됨
    # 7 -> 0~1은 항상 0과 1이 저장, 2는 얼굴 신뢰도가 저장, 3~6은 얼굴영역 사각형의 좌표가 0~1 사이로 정규화된 값

    for i in range(detect.shape[0]):
        confidence = detect[i, 2]    # 얼굴 신뢰도 값으로, 0~1사이의 값이며 1에 가까울수록 얼굴일 가능성이 큼
        if confidence < 0.5:         # N개의 얼굴 중 신뢰도가 임계값보다 낮으면 얼굴이 아니라고 봄
            break

        x1 = int(detect[i, 3] * w)
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

        label = 'Face: %4.3f' % confidence
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
