import cv2
import numpy as np

def image_draw(bbox, img):
    cv2.putText(img, 'People Counting : {}'.format(len(bbox)), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def detection_draw(bbox, landmark, img):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    if landmark is not None:
        for coord in landmark:
            cv2.circle(img, (int(coord[0]), int(coord[1])), 1, (0, 0, 255), -1)

def face_draw(bbox, img, name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if name == 'nobody':
        cv2.putText(img, '{}'.format(name), (int(bbox[0]), int(bbox[1]-10)), font, 0.6, (0, 0, 255), 2)  # font_size, Color, Thickness
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
    else:
        cv2.putText(img, '{}'.format(name), (int(bbox[0]), int(bbox[1]-10)), font, 0.6, (0, 255, 0), 2)