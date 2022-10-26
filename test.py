from deepface import DeepFace
import cv2

analysis = DeepFace.analyze(cv2.imread('__test__/michael.png'), actions=['race'])
print(analysis)