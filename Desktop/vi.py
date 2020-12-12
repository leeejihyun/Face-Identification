'''
FR 1:N(Identification) video version for face processing
'''
import time
import datetime
import os
import numpy as np
from numpy.linalg import norm
import insightface
import cv2
import model_loading as model
import face_cropping as crop
import face_recognition as recognition
import face_alignment as alignment
import face_draw as draw

class channel_info(): #camera channel
    def __init__(self):
        self.channel = None
        self.operate = None
        self.rtmp_path = None
        self.model = None
        self.test_id = None

        self.in_bytes = None
        self.image = None
        self.face_img = None

        self.bbox1 = None
        self.bbox = None
        self.bbox_list = []
        self.landmark1 = None
        self.landmark = None
        self.name_result = None

        self.track_id = []
        self.name_result_list = []
        self.name_result_list2 = []
        self.begin_end = []
        self.time = None

    def result_init(self):
        self.image = None
        self.face_img = None
        self.bbox1 = None
        self.bbox = None
        self.bbox_list = []
        self.landmark1 = None
        self.landmark = None
        self.name_result = None

        self.track_id = []
        self.name_result_list = []
        self.name_result_list2 = []
        self.begin_end = []
        self.time = None

def front_face(bboxs, landmarks):
	'''
	using landmark position, filter only fronted face cases
	:input param landmark:
	:return: filtered bbox, landmark
	'''
	idx = []
	for i, land in enumerate(landmarks):
		eye_dist = np.sqrt((land[1][0] - land[0][0]) ** 2 + (land[1][1] - land[0][1]) ** 2)
		# eye_xdist = land[1][0] - land[0][0]
		# eyelip_dist1 = np.sqrt((land[3][0] - land[0][0]) ** 2 + (land[3][1] - land[0][1]) ** 2)
		# eyelip_dist2 = np.sqrt((land[4][0] - land[1][0]) ** 2 + (land[4][1] - land[1][1]) ** 2)
		# angle = abs(np.arctan((land[1][1] - land[0][1]) / (land[1][0] - land[0][0])) * 180 / np.pi)
		eyenose_dist1 = np.sqrt((land[0][0] - land[2][0]) ** 2 + (land[0][1] - land[2][1]) ** 2)
		eyenose_dist2 = np.sqrt((land[1][0] - land[2][0]) ** 2 + (land[1][1] - land[2][1]) ** 2)
		lipnose_dist1 = np.sqrt((land[3][0] - land[2][0]) ** 2 + (land[3][1] - land[2][1]) ** 2)
		lipnose_dist2 = np.sqrt((land[4][0] - land[2][0]) ** 2 + (land[4][1] - land[2][1]) ** 2)

		if not(0.8 < (eyenose_dist1 / eyenose_dist2) < 1.2 and 0.8 < (lipnose_dist1 / lipnose_dist2) < 1.2) or eye_dist < 10:
			idx.append(i)
		else:
			pass

	bboxs = np.delete(bboxs, idx, 0)
	landmarks = np.delete(landmarks, idx, 0)
	return bboxs, landmarks

def filtering(bbox, landmark):
	# Filtering for high confidence
	del_list = [idx for idx, value in enumerate(bbox) if value[4] < 0.8]
	del_list.sort(reverse=True)
	for del_idx in del_list:
		bbox = np.delete(bbox, del_idx, 0)
		landmark = np.delete(landmark, del_idx, 0)
	
	# filtering for fronted face only
	if len(bbox) == 0:
		bbox1, landmark1 = [], []
	else:
		bbox1, landmark1 = front_face(bbox, landmark)
	return bbox1, landmark1

def main(folder, ch, detect_model, recognition_model, gallary_list, emb_list):
	path_list = os.listdir(folder)
	
	for i, path in enumerate(path_list):
		# Video File Open by VideoCapture
		ch.cap = cv2.VideoCapture('{}/{}'.format(folder, path))
		print('{}th video start'.format(i))

		# Class initialization
		ch.result_init()

		while True:
			# Read Frame
			for idx in range(5): # 5 : sampling rate, 1 frame from 5 frames, 6 frame per second
				ch.ret, ch.image = ch.cap.read() 
			if not ch.ret:
				break

			# Detection
			ch.bbox, ch.landmark = detect_model.detect(ch.image, threshold=0.5, scale=0.5) # Loose Case
			# ch.bbox1, ch.landmark1 = filtering(ch.bbox, ch.landmark)  # Tight Case

			# Recognition
			for box in ch.bbox: #Tight Case: ch.bbox1
				ch.face_img = crop.ver5(ch.image, box)
				ch.name_result = recognition.identification(ch.face_img, gallary_list, emb_list, recognition_model, 0.65)
				cv2.imshow('img', ch.face_img)
				draw.face_draw(box, ch.image, ch.name_result)
				cv2.imwrite('face_result/{}_{}.jpg'.format(i,time.time()), ch.face_img)

			# Overall Image Showing
			draw.detection_draw(ch.bbox, ch.landmark, ch.image)
			draw.image_draw(ch.bbox, ch.image)
			ch.image = cv2.resize(ch.image, None, fx=0.5, fy=0.5)
			cv2.imshow('frame', ch.image)
			cv2.waitKey(1)

if __name__ == "__main__":
	# Model Loading
	detect_model = model.detect_loading(0)
	recognition_model = model.recognition_loading(0)

	# gallary_list, emb_list Loading
	filename_npy = 'enroll.npz'
	gallary_list, emb_list = model.gallary_loading(filename_npy)

	# Input Video Folder
	folder = 'test_vid'

	# Class Open
	ch = channel_info()
	ch.test_id = 0
	ch.model = 'fourind'
	ch.channel = 0

	# Main
	main(folder, ch, detect_model, recognition_model, gallary_list, emb_list)
	print('k')
