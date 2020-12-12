import cv2
import numpy as np


def image_rotate(img, landmark):
	angle = np.arctan((landmark[0][1][1] - landmark[0][0][1]) / (landmark[0][1][0] - landmark[0][0][0])) * 180 / np.pi
	M1 = cv2.getRotationMatrix2D((112/2, 112/2), angle, 1)
	face_img = cv2.warpAffine(img, M1, (112, 112))
	return face_img

