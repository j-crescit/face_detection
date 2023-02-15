import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotate_math = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rotate_math, image.shape[1::-1], flags = cv2.INTER_LINEAR, borderValue = (255, 255, 255))
    return result

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:

            for detection in results.detections:

                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]
                nose_tip = keypoints[2]

                h, w, _ = image.shape
                right_eye = (int(right_eye.x * w) - 100, int(right_eye.y * h) - 150)
                left_eye = (int(left_eye.x * w) + 20, int(left_eye.y * h) - 150)
                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))
                
                mp_drawing.draw_detection(image, detection)

                tan_theta = (left_eye[1] - right_eye[1]) / (right_eye[0] - left_eye[0])
                theta = np.arctan(tan_theta)
                rotate_angle = theta * 180 / math.pi
                rotate_image_right_eye = rotate_image(image_right_eye, rotate_angle)
                rotate_image_left_eye = rotate_image(image_left_eye, rotate_angle)
                rotate_image_nose = rotate_image(image_nose, rotate_angle)

        cv2.imshow('Face detection', cv2.resize(image, None, fx=0.5, fy=0.5))

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()