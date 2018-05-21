import cv2
import dlib
import openface
import face_detection

predictor_model = "/home/unknownpgr/anaconda3/envs/opencv/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
face_aligner = openface.AlignDlib(predictor_model)

# Training part
model_path = "/home/unknownpgr/anaconda3/envs/opencv/openface/models/openface/nn4.small2.v1.t7"

with face_detection.FaceDetector(torch_net_model=model_path) as fd:
    fd.append_dir("Obama", "./training_images/obama_aligned")
    fd.append_dir("Trump", "./training_images/trump_aligned")
    fd.append_dir("Unknown", "./training_images/unknowns_aligned")

    fd.train_model()

    fd.save('./face_detector')

# Prediction part

with face_detection.FaceDetector.load('./face_detector') as fd:
    def get_frame_size(capture):
        return capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def draw_caption(img, rect, caption):
        cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 64))
        cv2.putText(img, caption, (rect.left(), rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 32))


    cv2.namedWindow('Window')
    cv2.moveWindow('Window', 20, 30)

    cap = cv2.VideoCapture()
    cap.open('./training_videos/vid_test.mp4')

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    size = get_frame_size(cap)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    out = cv2.VideoWriter('test_video.avi', fourcc, 30.0, (int(size[0] / 2), int(size[1] / 2)))

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(size[0] / 2), int(size[1] / 2)))
        for i, rect in enumerate(openface.AlignDlib.getAllFaceBoundingBoxes(frame)):
            aligned_image = face_aligner.align(534, frame, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            aligned_image = cv2.resize(aligned_image, (fd.img_dim, fd.img_dim))
            label = fd.predict(aligned_image)
            draw_caption(frame, rect, label)
        out.write(frame)

        cv2.imshow('Window', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
