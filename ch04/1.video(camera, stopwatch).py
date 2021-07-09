import numpy as np
import cv2


def camera_in():
    cap = cv2.VideoCapture(0)        # 카메라 지정하고 기기 작동 시작

    if not cap.isOpened():
        print("Camera open failed!")
        return

    print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))        # 프레임 정보 받기
    print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        ret, frame = cap.read()             # 프레임마다 영상 받아오기

        if not ret:
            break

        inversed = ~frame

        cv2.imshow('frame', frame)
        cv2.imshow('inversed', inversed)

        if cv2.waitKey(10) == 27:           # 10ms동안 ESC 키를 기다림. waitKey(0)이면 무한 대기. ms는 사실상 큰 의미 없음
            # 키가 눌려지지 않으면 리턴값은 -1이 되고 break가 실행되지 않고 다음으로 넘어감(while 문 계속 실행)
            break

    cv2.destroyAllWindows()


def video_in():
    cap = cv2.VideoCapture('stopwatch.avi')

    if not cap.isOpened():
        print("Video open failed!")
        return

    print('Frame width:', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('Frame height:', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print('Frame count:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print('FPS:', fps)

    delay = round(1000 / fps)       # 33ms를 기다림  -> 초당 30프레임을 만들기 위한 delay 값

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        inversed = ~frame

        cv2.imshow('frame', frame)
        cv2.imshow('inversed', inversed)

        if cv2.waitKey(delay) == 27:    # 초당 30프레임을 만들기 위한 기다림
            break

    cv2.destroyAllWindows()


def camera_in_video_out():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera open failed!")
        return

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D', 'I', 'V', 'X'
    # DIVX : avi 파일,  MJPG : mp4 파일, X264 : mkv 파일

    delay = round(1000 / fps)

    outputVideo = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

    if not outputVideo.isOpened():
        print('File open failed!')
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        inversed = ~frame

        outputVideo.write(inversed)

        cv2.imshow('frame', frame)
        cv2.imshow('inversed', inversed)

        if cv2.waitKey(delay) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera_in()
    video_in()
    camera_in_video_out()
