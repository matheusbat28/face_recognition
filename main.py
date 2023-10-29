import cv2
import mediapipe as mp


webcam = cv2.VideoCapture(0)
face_detector = mp.solutions.face_detection.FaceDetection()
drawing = mp.solutions.drawing_utils

while webcam.isOpened():
    ret, frame = webcam.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = face_detector.process(frame)
    if results.detections:
        for id, detection in enumerate(results.detections):
            drawing.draw_detection(frame, detection)

    cv2.imshow('Reconhecimento facial', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite('screenshot.jpg', frame)
