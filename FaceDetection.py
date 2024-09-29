import cv2
import matplotlib.pyplot as plt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def adjusted_detect_face(img):
    face_rect = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 10)
    return img

def detect_eye(img):
    eye_img = img.copy()
    eye_rect = eye_cascade.detectMultiScale(eye_img, scaleFactor = 1.3, minNeighbors = 5)
    for (x, y, w, h) in eye_rect:
        cv2.rectangle(eye_img, (x, y), (x+w, y+h), (255, 255, 255), 10)
    return eye_img
img1 = cv2.imread('abhishek.jpg')
img_copy3 = img1.copy()
eyes_face = adjusted_detect_face(img_copy3)
plt.imshow(eyes_face)
plt.show()