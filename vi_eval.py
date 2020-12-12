'''
FR 1:N(Identification) evaluation
FNIR @ FPIR of 0.01
'''
import os
import cv2
import utils.model_loading as model
import utils.face_cropping as crop
import utils.face_recognition as recognition
import utils.face_draw as draw
import matplotlib.pyplot as plt
import numpy as np

def main3(path, detect_model, recognition_model, subject_id_list, emb_list, split_num):
    
    if not os.path.exists('identification_result/split{}'.format(split_num)):
        os.makedirs('identification_result/split{}'.format(split_num))
    
    subject_id_list = []
    file_list = []
    gt_bbox_list = []
    fnir_list, fpir_list = [], []

    # load subject ID & file of probe
    with open(path+"/IJB-A_1N_sets/split{}/search_probe_{}.csv".format(split_num, split_num), 'r') as f:
        header = f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            line = line.rstrip()

            # subject ID
            subject_id = int(line.split(',')[1]) # SUBJECT_ID
            subject_id_list.append(subject_id)

            # file name
            file = line.split(',')[2] # FILE
            file_list.append(file)

            # ground truth bounding box
            face_x = float(line.split(',')[6]) # FACE_X
            face_y = float(line.split(',')[7]) # FACE_Y
            face_width = float(line.split(',')[8]) # FACE_WIDTH
            face_height = float(line.split(',')[9]) # FACE_HEIGHT
            gt_bbox = np.array([face_x, face_y, face_x + face_width, face_y + face_height])
            gt_bbox_list.append(gt_bbox)

    # calculate FNIR, FPIR for each similarity threshold
    threshold_list = [x / 10 for x in range(10, 0, -1)]  # 1.0 ~ 0.1
    scale_percent = 0.4
    for threshold in threshold_list:
        print('Threshold: {}'.format(threshold))

        if not os.path.exists('identification_result/split{}/{}'.format(split_num, threshold)):
            os.makedirs('identification_result/split{}/{}'.format(split_num, threshold))
        
        fpos, fneg = 0, 0  # false positive, false negative
        for i, file in enumerate(file_list):
            
            folder_name = file.split('/')[0]
            file_name = file.split('/')[1]
            
            # Image File Open
            image = cv2.imread('{}/{}'.format(path+'/images/', file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR보다 RGB일 때 detect 성능이 더 좋으므로 RGB로 변경
            print('Read the {}th image'.format(i))
            image = cv2.resize(image, None, fx=scale_percent, fy=scale_percent)

            # Detection
            bbox, landmark = detect_model.detect(image, threshold=0.5, scale=1.0)

            # Recognition
            face_img, bbox, landmark = crop.ver0(image, bbox, landmark, gt_bbox_list[i]*scale_percent)
            name_result = recognition.identification(face_img, subject_id_list, emb_list, recognition_model, threshold)
            if name_result is not 'nobody':
                name_result = subjectnames_by_id[name_result]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR) # imshow, imwrite할 땐 BGR로 다시 바꿔야 RGB로 보여지고 저장됨
            cv2.imshow('img', face_img)
            cv2.imwrite('identification_result/split{}/{}/{}_face_{}_{}_{}.jpg'.format(split_num, threshold, i, folder_name, file_name, name_result), face_img)
            draw.face_draw(bbox, image, name_result)

            if name_result == 'nobody':
                # false negative: 등록을 미등록으로 인식
                if subject_id_list[i] in subject_id_list:
                    fneg += 1
            else:
                # false positive: 미등록을 등록으로 인식
                if subject_id_list[i] not in subject_id_list:
                    fpos += 1

            # Overall Image Showing
            draw.detection_draw(bbox, landmark, image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow('image', image)
            cv2.imwrite('identification_result/split{}/{}/{}_{}_{}_{}.jpg'.format(split_num, threshold, i, folder_name, file_name, name_result), image)
            cv2.waitKey(1)

            if i == len(file_list)-1 :
                print("-"*30)
                break

        # FNIR: False Negative Identification Rate, FPIR: False Positive Identification Rate
        fnir = fneg / len(file_list)
        fpir = fpos / len(file_list)
        fnir_list.append(fnir)
        fpir_list.append(fpir)
        
        print("Threshold: {}, FNIR @ FPIR of {}: {}".format(threshold, round(fpir,2), round(fnir,2)))
        with open("result_threshold{}_split{}.txt".format(threshold, split_num), 'w') as f:
            line = "Threshold: {}, FNIR @ FPIR of {}: {}\n".format(threshold, round(fpir,2), round(fnir,2))
            f.write(line)
        if round(fpir, 2) == 0.01:
            break

    # draw DET Curve
    plt.scatter(fpir_list, fnir_list)
    plt.plot(fpir_list, fnir_list)
    plt.grid()
    plt.xlabel("FPIR(False Positive Identification Rate")
    plt.ylabel("False Negative Identification Rate")
    plt.title("DET Curve")
    plt.savefig("identification_result/DET Curve_split{}.png".format(split_num))

if __name__ == "__main__":
    # Model Loading
    detect_model = model.detect_loading(0)
    recognition_model = model.recognition_loading(0)

    # Dataset Path
    path = 'IJB-A'
    
    # Make dictionary to get names by id
    subjectnames_by_id = {}
    with open(path + "/IJB-A_subjectnames_by_id.csv", 'r') as f:
        header = f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            line = line.rstrip()
            
            subject_id = int(line.split(',')[0])
            name = line.split(',')[1]
            subjectnames_by_id[subject_id] = name
    
    if not os.path.exists('identification_result'):
        os.makedirs('identification_result')
    
    # subject_id_list, emb_list Loading
    for split_num in range(5, 11):
        
        print("split{}".format(split_num))
        
        filename_npy = 'enroll_split{}.npz'.format(split_num)
        subject_id_list, emb_list = model.gallary_loading(filename_npy)

        # Main
        main3(path, detect_model, recognition_model, subject_id_list, emb_list, split_num)
        print('Finished!')