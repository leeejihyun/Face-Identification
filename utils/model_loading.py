import insightface
import insightface_new
import numpy as np
import cv2
from io import BytesIO as bi
from PIL import Image
import requests
import ffmpeg
import time

def detect_loading(num):
	# Model Loading : Deteciton and Recognition
# 	face_detect_model = insightface.model_zoo.get_model('retinaface_r50_v1')
	face_detect_model = insightface_new.model_zoo.get_model('retinaface_mnet025_v2')
	face_detect_model.prepare(ctx_id = num, nms=0.4)
	return face_detect_model

def recognition_loading(num):
	face_recognition_model = insightface.model_zoo.get_model('arcface_r100_v1')
	face_recognition_model.prepare(ctx_id = num)
	return face_recognition_model

def gallary_loading(filename):
	enrol_list = np.load(filename)
	gallary_list = enrol_list['x']
	emb_list = enrol_list['y']
	enrol_list.close()
	print('Gallary Preparation Complete!')
	return gallary_list, emb_list 

def npy_loading(face, filename_npy, npy_file_load):
	if npy_file_load == True:
		name_list, emb_list = gallary_loading(filename_npy)
		face.name_list = name_list.tolist()
		face.emb_list = emb_list.tolist()
	else:
		face.name_list = []
		face.emb_list = []
	return face

def resize_img(img, img_min_side=600):
    height, width  = img.shape[:2]
    if width >= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)
    return cv2.resize(img, (resized_width, resized_height))