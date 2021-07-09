import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/opencv/ch16/2.dnnmnist')
import sys
import numpy as np
import cv2


filename = 'space_shuttle.jpg'

if len(sys.argv) > 1:
    filename = sys.argv[1]

img = cv2.imread(filename)

if img is None:
    print('Image load failed!')
    sys.exit()

# Load network

net = cv2.dnn.readNet('bvlc_googlenet.caffemodel', 'deploy.prototxt')

if net.empty():
    print('Network load failed!')
    sys.exit()

# Load class names

classNames = None
with open('classification_classes_ILSVRC2012.txt', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Inference

inputBlob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))  
# 카페에서 학습된 구글넷은 입력으로 224×224 크기의 BGR 컬러 영상을 사용, 각 영상에서 평균값 Scalar(104, 117, 123)을 빼서 학습
# 결과로 나온 inputBlob 행렬은 1×3×224×224 형태를 갖는 4차원 행렬

net.setInput(inputBlob, 'data')
prob = net.forward()    # 1×1000 사이즈의 행렬

# Check results & Display

out = prob.flatten()
classId = np.argmax(out)   # 최대값 추출
confidence = out[classId]

str = '%s (%4.2f%%)' % (classNames[classId], confidence * 100)
cv2.putText(img, str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
