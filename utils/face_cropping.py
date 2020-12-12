import numpy as np
import cv2
from skimage import transform as trans

def ver5(img, bbox, option=3):
    bbox = bbox.astype(np.int).flatten()
    f_box= [0,0,0,0]
    if option == 1:
        expand_ratio = 0.15
        f_box[0] = max(1, int((1-expand_ratio)*bbox[0]))
        f_box[1] = max(1, int((1-expand_ratio)*bbox[1]))
        f_box[2] = int((1+expand_ratio)*bbox[2])
        f_box[3] = int((1+expand_ratio)*bbox[3])
    elif option == 2:
        f_box[0] = max(1, int(bbox[0])-10)
        f_box[1] = max(1, int(bbox[1])-20)
        f_box[2] = int(bbox[2]+10)
        f_box[3] = int(bbox[3]+10)
    elif option == 3:
        expand_ratio = 0.2
        center_x = int(0.5 * (bbox[0]+bbox[2]))
        center_y = int(0.5 * (bbox[1]+bbox[3]))
        length = int(0.5*(1+expand_ratio)*(bbox[3]-bbox[1]))
        f_box = [center_x-length, center_y-length, center_x+length, center_y+length]
    elif option == 4:
        expand_ratio = 0.3
        center_x = int(0.5 * (bbox[0]+bbox[2]))
        center_y = int(0.5 * (bbox[1]+bbox[3]))
        length = int(0.5*(1+expand_ratio)*(bbox[2]-bbox[0]))
        f_box = [center_x-length, center_y-length, center_x+length, center_y+length]
    face_img = img[max(1, f_box[1]):f_box[3], max(1, f_box[0]):f_box[2]]
    face_img = cv2.resize(face_img, (112, 112))
    return face_img

def ver6(img, bboxs, expand_ratio=0.1):
    h, w = img.shape[:2]
    bboxs[bboxs<0] = 0
    bboxs[bboxs[:,2]>w, 2] = w
    bboxs[bboxs[:,3]>h, 3] = h
    
    img_center = np.array([img.shape[1]/2, img.shape[0]/2])
    center_x =(bboxs[:,0]+bboxs[:,2])/2
    center_y = (bboxs[:,1]+bboxs[:,3])/2
    center_bbox = np.vstack([center_x, center_y]).T
    l2_dists = np.sqrt(np.square(img_center-center_bbox).sum(axis=1))
    
    length_w = (bboxs[:,2]-bboxs[:,0])
    length_h = (bboxs[:,3]-bboxs[:,1])
    areas = length_w*length_h
    
    confidences = bboxs[:,4]
    
    eval_bbox = np.argsort(-l2_dists)*0.4 + np.argsort(areas)*0.5 + np.argsort(confidences)*0.1
    idx = np.argmax(eval_bbox)
    
    new_x1 = center_x[idx] - (length_w[idx]*(1+expand_ratio)/2)
    new_x2 = center_x[idx] + (length_w[idx]*(1+expand_ratio)/2)
    new_y1 = center_y[idx] - (length_h[idx]*(1+expand_ratio)/2)
    new_y2 = center_y[idx] + (length_h[idx]*(1+expand_ratio)/2)
    
    if new_x1<0: new_x1 = 0
    if new_y1<0: new_y1 = 0
    if new_x2>w: new_x2 = w
    if new_y2>h: new_y2 = h
        
    face_img = img[int(new_y1):int(new_y2), int(new_x1):int(new_x2), :]
    return cv2.resize(face_img, (112, 112))

def heuristic(img):
    face_img = img[60:190, 60:190]
    face_img = cv2.resize(face_img, (112, 112))
    return face_img

def landmark_transform(bbox, landmark):
    bbox = bbox.flatten()
    x0 = bbox[0]
    y0 = bbox[1]
    for i in range(5):
        landmark[i][0] -= x0
        landmark[i][1] -= y0
    box_len_x = bbox[2]-bbox[0]
    box_len_y = bbox[3]-bbox[1]
    for i in range(5):
        landmark[i][0] /= (box_len_x/112)
        landmark[i][1] /= (box_len_y/112)
    dst = landmark.astype(np.float32)
    return dst

def ver1(img, bbox):
    expand_ratio = -0.01
    f_box= [0,0,0,0]
    height, width = img.shape[0], img.shape[1]
    b_num = bbox.shape[0]

    bbox = bbox.astype(np.int)
    center_x = int(0.5 * (bbox[0] + bbox[2]))
    center_y = int(0.5 * (bbox[1] + bbox[3]))
    length_x = int(0.5 * (1 + expand_ratio) * (bbox[2] - bbox[0]))
    length_y = int(0.5 * (1 + expand_ratio) * (bbox[3] - bbox[1]))
    f_box = [center_x - length_x, center_y - length_y, center_x + length_x, center_y + length_y]
    face_img = img[max(1, f_box[1]):f_box[3], max(1, f_box[0]):f_box[2]]
            
    if f_box == [0, 0, 0, 0]:
        print('Face is not centered or not detected')
        center_x = int(0.5 * width)
        center_y = int(0.5 * height)
        length_x = int(0.25 * width)
        length_y = int(0.25 * height)       
        f_box = [center_x - length_x, center_y - length_y, center_x + length_x, center_y + length_y]
        f_box = np.array(f_box)
        f_box[f_box<0]=0
        if f_box[2]>width:
            f_box[2]=width
        if f_box[3]>height:
            f_box[3]=height
        face_img = img[max(1, f_box[1]):f_box[3], max(1, f_box[0]):f_box[2]]
    face_img = cv2.resize(face_img, (112, 112))
    return face_img

def ver0(img, bbox=None, landmark=None, gt_bbox=None, margin=44):    
    
    input_height, input_width = img.shape[0], img.shape[1]
    output_height, output_width = 112, 112
    center_x = input_width/2
    center_y = input_height/2
    M = None
    b_num = bbox.shape[0]
    
    # bounding box가 없을 경우
    if b_num == 0:
        bbox = None
        landmark = None
        
    # bounding box가 1개일 경우
    elif b_num == 1:
        bbox = bbox[0]
        try:landmark = landmark[0]
        except:pass
    
    # bounding box가 여러개일 경우
    else:       
        # 데이터셋에서 제공하는 수동 bounding box와 가장 가까운 bounding box 선택
        dist_list = []
        for b_idx in range(b_num):
            gt_b_center = np.array(((gt_bbox[0] + gt_bbox[2])*0.5, (gt_bbox[1] + gt_bbox[3])*0.5))
            b_center = np.array(((bbox[b_idx][0] + bbox[b_idx][2])*0.5, (bbox[b_idx][1] + bbox[b_idx][3])*0.5))
            dist = np.linalg.norm(gt_b_center - b_center)
            dist_list.append(dist)
        choice = dist_list.index(min(dist_list))
        bbox = bbox[choice]
        try:landmark = landmark[choice]
        except:pass
        
    # landmark가 있을 경우 similarity transformation
    if landmark is not None:
        src = np.array([
            [38.2946, 47.6963],
            [73.5318, 47.5014],
            [56.0252, 67.7366],
            [41.5493, 88.3655],
            [70.7299, 88.2041] ], dtype=np.float32)

        dst = landmark_transform(bbox, landmark)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]

    # landmark가 없을 경우
    if M is None:
        
        # bounding box가 없으면 데이터셋에서 제공하는 수동 bounding box 선택
        if bbox is None: 
            print('The face is not detected. Use ground truth bounding box.')
            bbox = gt_bbox
            
        # margin을 이용해 crop 영역 지정
        # crop하고 이미지 사이즈 변경
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(bbox[0]-margin/2, 0)
        bb[1] = np.maximum(bbox[1]-margin/2, 0)
        bb[2] = np.minimum(bbox[2]+margin/2, input_width)
        bb[3] = np.minimum(bbox[3]+margin/2, input_height)
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        ret = cv2.resize(ret, (output_height, output_width))
        
        bbox = bbox.astype(np.int)
        
        return ret, bbox, landmark
    
    # landmark가 있을 경우
    else:
        warped = cv2.warpAffine(ver1(img, bbox), M, (output_height, output_width), borderValue = 0.0)
        bbox = bbox.astype(np.int)
        landmark = landmark.astype(np.int)
        return warped, bbox, landmark