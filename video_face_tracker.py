import openface
import cv2
import dlib
import numpy as np

face_detector = dlib.get_frontal_face_detector()


class FaceInfo:
    # 얼굴에 달아줄 인덱스.
    face_index = 0

    # 검출된 얼굴 리스트.
    face_list = []

    def __init__(self, rect):
        self.rect = rect
        # checked 변수는 몇 프레임 전에 업데이트되었는지 체크함.
        self.checked = 0
        self.index = FaceInfo.face_index
        # 화면에 보일지를 결정함.
        self.visible = True
        FaceInfo.face_index += 1

    # 현재 검출된 영역과 new_rect가 가까운지 체크함.
    def is_near(self, new_rect):
        return FaceInfo.get_dist(FaceInfo.get_center(self.rect), FaceInfo.get_center(new_rect)) < FaceInfo.get_r(
            self.rect)

    # 얼굴 위치를 업데이트함.
    def update(self, new_rect):
        self.rect = new_rect
        self.checked = 0

    # 얼굴 영역의 좌상단과 우하단의 위치를 반환함.
    def rect_points(self):
        return (self.rect.left(), self.rect.top()), (self.rect.right(), self.rect.bottom())

    # 두 위치 사이의 거리를 구함.
    @staticmethod
    def get_dist(posA, posB):
        return np.linalg.norm(np.subtract(posA, posB))

    # 얼굴 영역의 반지름, 즉 사각형의 한 변의 절반을 구함.
    @staticmethod
    def get_r(rect):
        return (rect.right() - rect.left()) / 2

    # 얼굴 영역의 중심을 반환함.
    @staticmethod
    def get_center(rect):
        return np.array([(rect.right() + rect.left()) / 2, (rect.top() + rect.bottom()) / 2])

    # 주어진 영역들을 체크함.
    @staticmethod
    def check_face(faces):

        # 검출된 얼굴들.
        temp_list = []

        for i, rect in enumerate(faces):
            contained = False
            for face in FaceInfo.face_list:
                if face.is_near(rect):
                    contained = True
                    face.update(rect)
                    break
            if not contained:
                temp_list.append(FaceInfo(rect))

        for face in FaceInfo.face_list:
            if face.checked < 10:
                # checked 변수가 10보다 작아야만 검출된 것으로 판정함.
                temp_list.append(face)

            if face.checked is 0:
                face.visible = True
            else:
                face.visible = False

            face.checked += 1

        FaceInfo.face_list = temp_list

    @staticmethod
    # 프레임 위에 현재 검출된 영역들을 그림.
    def draw_rect(frame):
        for face in FaceInfo.face_list:
            if face.visible:
                points = face.rect_points()
                cv2.rectangle(frame, points[0], points[1], (0, 0, 255), 2)
                cv2.putText(frame, 'index = ' + str(face.index), (points[0][0], points[0][1] - 10), font, 0.8,
                            (0, 0, 255))
        return frame


def get_frame_size(capture):
    return capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Test part

video_path = "./test_video.mp4"
font = cv2.FONT_HERSHEY_SIMPLEX

capture = cv2.VideoCapture()
capture.open(video_path)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
size = get_frame_size(capture)
count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
# 속도를 위해서 영상 크기를 반으로 줄여 저장함.
out = cv2.VideoWriter('test_video.avi', fourcc, 20.0, (int(size[0] / 2), int(size[1] / 2)))

processed = 0

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (int(size[0] / 2), int(size[1] / 2)))

    faces = face_detector(frame, 1)
    FaceInfo.check_face(faces)
    frame = FaceInfo.draw_rect(frame)

    out.write(frame)

    rate = processed * 100 / count
    print("{0:.2f}".format(rate), " = ", processed, "/", count)

    processed += 1

capture.release()

print('finished')
