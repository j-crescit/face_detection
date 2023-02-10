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

def overlay(image, x, y, w, h, overlay_image):
    alpha = overlay_image[:, :, 3]
    mask_image = alpha / 255

    for c in range(0, 3):
        image[y - h : y + h, x - w : x + w, c] = (overlay_image[:, :, c] * mask_image) + (image[y - h : y + h, x - w : x + w, c] * (1 - mask_image))

cap = cv2.VideoCapture('video/face_video.mp4')

image_right_eye = cv2.imread('image/left_eye.png', cv2.IMREAD_UNCHANGED)
image_left_eye = cv2.imread('image/right_eye.png', cv2.IMREAD_UNCHANGED)
image_nose = cv2.imread('image/nose.png', cv2.IMREAD_UNCHANGED)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:

    while cap.isOpened():
      success, image = cap.read()
      if not success:
        break

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

          image[right_eye[1] - 50 : right_eye[1] + 50, right_eye[0] - 50 : right_eye[0] + 50] = image_right_eye
          image[left_eye[1] - 50 : left_eye[1] + 50, left_eye[0] - 50 : left_eye[0] + 50] = image_left_eye
          image[nose_tip[1] - 50 : nose_tip[1] + 50, nose_tip[0] - 150 : nose_tip[0] + 150] = image_nose

          tan_theta = (left_eye[1] - right_eye[1]) / (right_eye[0] - left_eye[0])
          theta = np.arctan(tan_theta)
          rotate_angle = theta * 180 / math.pi
          rotate_image_right_eye = rotate_image(image_right_eye, rotate_angle)
          rotate_image_left_eye = rotate_image(image_left_eye, rotate_angle)
          rotate_image_nose = rotate_image(image_nose, rotate_angle)

          overlay(image, *right_eye, 50, 50, rotate_image_right_eye)
          overlay(image, *left_eye, 50, 50, rotate_image_left_eye)
          overlay(image, *nose_tip, 150, 50, rotate_image_nose)

      cv2.imshow('Face Detection', cv2.resize(image, None, fx = 0.5, fy =0.5))

      if cv2.waitKey(1) == ord('q'):
        break
      
cap.release()
cv2.destroyAllWindows()