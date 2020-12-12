# FR 1:N(Identification) video version
import insightface
import cv2
import numpy as np
import time
import datetime
import csv
import os
from numpy.linalg import norm

import model_loading as model
import face_cropping as crop
import face_recognition as recognition
import face_alignment as alignment
import face_draw as draw
import buildJson

class channel_info():
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
        self.name_result_list2 = []
        self.begin_end = []
        self.time = None

def front_face(bboxs, landmarks):
    idx = []
    for i, land in enumerate(landmarks):
        bbox_xdist = bboxs[i][2]-bboxs[i][0]
        eye_dist = np.sqrt((land[1][0] - land[0][0]) ** 2+(land[1][1] - land[0][1]) ** 2)
        eye_xdist = land[1][0] - land[0][0]
        eyelip_dist1 = np.sqrt((land[3][0] - land[0][0]) ** 2+(land[3][1] - land[0][1]) ** 2)
        eyelip_dist2 = np.sqrt((land[4][0] - land[1][0]) ** 2 + (land[4][1] - land[1][1]) ** 2)
        eyenose_dist1 = np.sqrt((land[0][0] - land[2][0]) ** 2 + (land[0][1] - land[2][1]) ** 2)
        eyenose_dist2 = np.sqrt((land[1][0] - land[2][0]) ** 2 + (land[1][1] - land[2][1]) ** 2)
        lipnose_dist1 = np.sqrt((land[3][0] - land[2][0]) ** 2 + (land[3][1] - land[2][1]) ** 2)
        lipnose_dist2 = np.sqrt((land[4][0] - land[2][0]) ** 2 + (land[4][1] - land[2][1]) ** 2)
        #print(bbox_xdist, eye_xdist)
        if bbox_xdist < 32 or eye_xdist/bbox_xdist < 0.22:
            idx.append(i)
        elif not (0.75 < (eyenose_dist1 / eyenose_dist2) < 1.3 and 0.75 < (lipnose_dist1 / lipnose_dist2) < 1.3):
            idx.append(i)
        elif not (0.75 < (eyenose_dist1 / lipnose_dist1) < 1.3 and 0.75 < (eyenose_dist2 / lipnose_dist2) < 1.3):
            idx.append(i)
        else:
            pass
    bboxs = np.delete(bboxs, idx, 0)
    landmarks = np.delete(landmarks, idx, 0)
    return bboxs, landmarks

def intersection_over_union(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # BBOX 가로,세로 +10
    if (x2 - x1 + 10) < 0 or (y2 - y1 + 10) < 0 :
        # bbox1, bbox2가 겹치지 않는 경우
        iou = 0
    else :
        inter_area = (x2 - x1 + 10) * (y2 - y1 + 10)
        bbox1_area = (bbox1[2] - bbox1[0] + 10) * (bbox1[3] - bbox1[1] + 10)
        bbox2_area = (bbox2[2] - bbox2[0] + 10) * (bbox2[3] - bbox2[1] + 10)
        union_area = bbox1_area + bbox2_area - inter_area
        iou = inter_area / union_area

    return iou

def same_bbox(bbox1, bbox2, num=2):
    bbox1 = bbox1.flatten()
    bbox2 = bbox2.flatten()

    if intersection_over_union(bbox1, bbox2) > 0 :
        return True
    else :
        if len(ch.bbox) == 1:
            # People Counting = 1
            return True
        else:
            return False

    # x1 = int(0.5 * (bbox1[0] + bbox1[2]))
    #     # y1 = int(0.5 * (bbox1[1] + bbox1[3]))
    #     # len1 = bbox1[2]- bbox1[0]
    #     #
    #     # x2 = int(0.5 * (bbox2[0] + bbox2[2]))
    #     # y2 = int(0.5 * (bbox2[1] + bbox2[3]))
    #     # len2 = bbox2[2]- bbox2[0]
    #     # dist1 = np.sqrt((x1-x2)**2+(y1-y2)**2)
    #     # dist2 = 0.5*(len1+len2)
    #     # #print(dist2/dist1)
    #     #
    #     # if dist1 < num*dist2:
    #     #     return True
    #     # else :
    #     #     return False

def recognize_process(ch, fr, recognition_model, gallary_list, emb_list):
    N_bbox = len(ch.bbox)
    N_begin = ch.begin_end.count('begin')
    draw.image_draw(ch.bbox, ch.image)
    if N_begin == 0:
        if N_bbox > 0:
            ch.bbox, ch.landmark = filtering(ch.bbox, ch.landmark)
            if len(ch.bbox) > 0:
                for bbox in ch.bbox:
                    ch = recognize_begin(ch, fr, bbox, gallary_list, emb_list, recognition_model)
                draw.detection_draw(ch.bbox, ch.landmark, ch.image)


    elif N_begin > 0:
        # 기존 외에 새로 생기는 CASE : ch.bbox마다 체크
        for idx1, bbox1 in enumerate(ch.bbox):
            # SAME BOX CHECK : same box가 있으면 새롭게 좌표 update, 없으면 False List 추가, 모두 없는 경우면 새로운 BBOX라는 뜻
            False_list = [1] * len(ch.bbox_list)
            for idx2, bbox2 in enumerate(ch.bbox_list):
                False_list[idx2] = same_bbox(bbox1, bbox2)
                if False_list[idx2] == True:
                    ch.bbox_list[idx2] = bbox1

            # NEW BOX CHECK : filtering해도 존재하는 경우, begin으로 시작
            if False_list == [False] * len(ch.bbox_list):
                ch.bbox1, ch.landmark1 = filtering(ch.bbox, ch.landmark)
                if bbox1 in ch.bbox1:
                    recognize_begin(ch, fr, ch.bbox[idx1], gallary_list, emb_list, recognition_model)

        # 기존에서 사라지는 생기는 CASE : ch.bbox_list마다 체크
        for idx2, bbox2 in reversed(list(enumerate(ch.bbox_list))):
            False_list = [1] * N_bbox
            for idx1, bbox1 in enumerate(ch.bbox):
                False_list[idx1] = same_bbox(bbox1, bbox2)

            if False_list == [False] * N_bbox:
                begin_idx = [idxx for idxx, value in enumerate(ch.begin_end) if value == 'begin']
                recognize_end(ch, fr, begin_idx[idx2])
                ch.bbox_list.pop(idx2)
        draw.detection_draw(ch.bbox, ch.landmark, ch.image)
    return ch

def filtering(bbox, landmark):
    del_list = [idx for idx, value in enumerate(bbox) if value[4] < 0.8]
    del_list.sort(reverse=True)
    for del_idx in del_list:
        bbox = np.delete(bbox, del_idx, 0)
        landmark = np.delete(landmark, del_idx, 0)
    if len(bbox) ==0:
        bbox1, landmark1 = [], []
    else:
        bbox1, landmark1 = front_face(bbox, landmark)
    return bbox1, landmark1

def recognize_begin(ch, fr, bbox, gallary_list, emb_list, recognition_model):
    ch.face_img = crop.ver5(ch.image, bbox)
    cv2.imshow('img', ch.face_img)
    ch.name_result = recognition.identification(ch.face_img, gallary_list, emb_list, recognition_model, 0.65)
    ch.name_result_list2.append(ch.name_result)
    ch.track_id.append(len(ch.track_id))
    ch.begin_end.append('begin')
    ch.bbox_list.append(bbox)
    ch.time = fr
    result = buildJson.vi(ch.model, ch.test_id, ch.channel, ch.time, ch.track_id[-1], 'begin', ch.name_result)
    print(result)
    return ch

def recognize_end(ch, fr, idx):
    ch.begin_end[idx] = 'end'
    ch.time = fr
    result = buildJson.vi(ch.model, ch.test_id, ch.channel, ch.time, ch.track_id[idx], 'end', ch.name_result_list2[idx])
    print(result)
    return ch



def main(folder, ch, detect_model, recognition_model, gallary_list, emb_list):
    path_list = os.listdir(folder)
    for i, path in enumerate(path_list):
        #print(i, path)
        ch.result_init()
        ch.model = path
        ch.test_id = i
        ch.cap = cv2.VideoCapture('{}/{}'.format(folder, path))
        fr = 0
        while True:
            for idx in range(5):
                ch.ret, ch.image = ch.cap.read()
            if not ch.ret:
                break

            # Main Code
            ch.bbox, ch.landmark = detect_model.detect(ch.image, threshold=0.45, scale=0.5)
            ch = recognize_process(ch, fr, recognition_model, gallary_list, emb_list)

            # Image Show
            ch.image = cv2.resize(ch.image, None, fx=0.5, fy=0.5)
            cv2.imshow('frame', ch.image)
            cv2.waitKey(1)
            fr += 5

if __name__ == "__main__":
    # Model Loading
    detect_model = model.detect_loading(0)
    recognition_model = model.recognition_loading(0)
    filename_npy = 'enroll4.npz'
    gallary_list, emb_list = model.gallary_loading(filename_npy)

    # Input Video
    folder = 'fr_strange/two'
    # folder = 'fr_strange/two/debugvid'
    model_name = 'fourind'
    test_id = 0

    ch = channel_info()
    ch.test_id = test_id
    ch.model = model_name
    ch.channel = 0
    # Main
    main(folder, ch, detect_model, recognition_model, gallary_list, emb_list)
    print('k')