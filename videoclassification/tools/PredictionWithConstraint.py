import torch
import os
import os.path as osp
import re
from operator import itemgetter
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
from mmaction.apis import init_recognizer
from mmcv.runner import set_random_seed
from mmcv import Config
from mmaction.apis import single_gpu_test
from mmaction.datasets import build_dataloader
from mmcv.parallel import MMDataParallel
from mmaction.models import build_model
from mmcv import Config
from mmaction.core import OutputHook
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import numpy as np
import time
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

def directory_classifier_restrict_1(data_dir, num_class = 15):
    txt_constrict = 'Classifier_restrict_vector_1st.txt'
    txt_name_constrict = os.path.join(data_dir, txt_constrict)
    constrict = open(txt_name_constrict, 'r')
    lines = constrict.readlines()
    cluster_dict = dict()
    for line in lines:
        line_split = line.split('\n')[0].split(' ')
        cluster = line_split[0]
        #Key generation
        key = cluster
        # One-hot label
        one_hot_label = np.zeros((num_class))
        for ele in line_split[1:]:
            one_hot_label[int(ele)] = 1

        cluster_dict[key] = one_hot_label
    return cluster_dict
        
def directory_classifier_restrict_2(data_dir, num_class = 62):
    txt_constrict = 'Classifier_restrict_vector_2nd.txt'
    txt_name_constrict = os.path.join(data_dir, txt_constrict)
    constrict = open(txt_name_constrict, 'r')
    lines = constrict.readlines()
    cluster_dict = dict()
    for line in lines:
        line_split = line.split('\n')[0].split(' ')
        #Key generation
        key = '{}{}'.format(line_split[0],line_split[1])
        # One-hot label
        one_hot_label = np.zeros((num_class))
        for ele in line_split[2:]:
            one_hot_label[int(ele)] = 1

        cluster_dict[key] = one_hot_label
    return cluster_dict

def directory_classifier_restrict_3(data_dir, num_class = 186):
    txt_constrict = 'Classifier_restrict_vector_3rd.txt'
    txt_name_constrict = os.path.join(data_dir, txt_constrict)
    constrict = open(txt_name_constrict, 'r')
    lines = constrict.readlines()
    cluster_dict = dict()
    for line in lines:
        line_split = line.split('\n')[0].split(' ')
        #Key generation
        key = '{}{}{}'.format(line_split[0],line_split[1],line_split[2])
        # One-hot label
        one_hot_label = np.zeros((num_class))
        for ele in line_split[3:]:
            one_hot_label[int(ele)] = 1

        cluster_dict[key] = one_hot_label
    return cluster_dict

def directory_classifier_restrict_4(data_dir, num_class = 179):
    txt_constrict = 'Classifier_restrict_vector_4th.txt'
    txt_name_constrict = os.path.join(data_dir, txt_constrict)
    constrict = open(txt_name_constrict, 'r')
    lines = constrict.readlines()
    cluster_dict = dict()
    for line in lines:
        line_split = line.split('\n')[0].split(' ')
        #Key generation
        key = '{}{}{}{}'.format(line_split[0],line_split[1],line_split[2],line_split[3])
        # One-hot label
        one_hot_label = np.zeros((num_class))
        for ele in line_split[4:]:
            one_hot_label[int(ele)] = 1

        cluster_dict[key] = one_hot_label
    return cluster_dict

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
    
def inference_recognizer(model,
                         video_path,
                         use_frames=False,
                         outputs=None,
                         as_tensor=True):
    if isinstance(outputs, str):
        outputs = (outputs, )
    assert outputs is None or isinstance(outputs, (tuple, list))

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    test_pipeline = Compose(test_pipeline)

    modality = cfg.data.test.get('modality', 'RGB')

    num_frames = len(os.listdir(video_path))

    data = dict(
        frame_dir=video_path,
        total_frames=num_frames,
        label=-1,
        start_index=0,
        filename_tmpl=None,
        modality=modality)

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    imgs = data['imgs']
    # forward the model
    with torch.no_grad():
        scores, scores_pro = model.forward_test_with_constraint(imgs)
    return scores, scores_pro

if __name__ == "__main__":
    data_dir = '/datasets/carAccident_videoClassification'
    log_dir = '/datasets/carAccident_videoClassification/0315'
    txt_name = 'test_1st.txt' # Classifier 1
    #txt_name = 'test_2nd.txt' # Classifier 2
    #txt_name = 'test_3rd.txt' # Classifier 3
    #txt_name = 'test_4th.txt' # Classifier 4

    config = '/datasets/carAccident_videoClassification/VIEW-3/For1/slowfast_r50_4x16x1_256e_kinetics400_rgb.py' # Classifier 1
    #config = '/datasets/carAccident_videoClassification/VIEW-3/For2/slowfast_r50_4x16x1_256e_kinetics400_rgb.py' # Classifier 2
    #config = '/datasets/carAccident_videoClassification/VIEW-3/For3/slowfast_r50_4x16x1_256e_kinetics400_rgb.py' # Classifier 3
    #config = '/datasets/carAccident_videoClassification/VIEW-3/For4/slowfast_r50_4x16x1_256e_kinetics400_rgb.py' # Classifier 4

    checkpoint = '/datasets/carAccident_videoClassification/VIEW-3/For1/epoch_20.pth' # Classifier 1
    #checkpoint = '/datasets/carAccident_videoClassification/VIEW-3/For2/epoch_20.pth' # Classifier 2
    #checkpoint = '/datasets/carAccident_videoClassification/VIEW-3/For3/epoch_20.pth' # Classifier 3
    #checkpoint = '/datasets/carAccident_videoClassification/VIEW-3/For4/epoch_30.pth' # Classifier 4

    cfg = Config.fromfile(config)

    cluster_dict = directory_classifier_restrict_1(data_dir) # Classifier 1
    #cluster_dict = directory_classifier_restrict_2(data_dir) # Classifier 2
    #cluster_dict = directory_classifier_restrict_3(data_dir) # Classifier 3
    #cluster_dict = directory_classifier_restrict_4(data_dir) # Classifier 4
    # Initialize the recognizer
    model = init_recognizer(config, checkpoint, device='cuda:0', use_frames=True)
    txt_name = os.path.join(data_dir, txt_name)

    result_log_name = os.path.join(log_dir, 'matrixlog_1st.txt')
    #result_log_name = os.path.join(log_dir, 'matrixlog_2nd.txt') 
    #result_log_name = os.path.join(log_dir, 'matrixlog_3rd.txt')
    #result_log_name = os.path.join(log_dir, 'matrixlog_4th.txt')
    txt_matrix = open(result_log_name, 'w')


    txt_car = open(txt_name, 'r')
    lines = txt_car.readlines()

    labels = []
    preds = []
    processing = 1
    total_sample = len(lines)
    for line in lines:
        try:
            patch_name = line.split('\n')[0].split(' ')
            label = int(patch_name[1])
            cluster = patch_name[2]
            key= '{}'.format(patch_name[2]) # Classifier 1
            #key= '{}{}'.format(patch_name[2], patch_name[3]) # Classifier 2
            #key= '{}{}{}'.format(patch_name[2], patch_name[3], patch_name[4]) # Classifier 3
            #key= '{}{}{}{}'.format(patch_name[2], patch_name[3], patch_name[4], patch_name[5]) # Classifier 4
            # print(line)
            block_path = os.path.join(data_dir, patch_name[0])
            if key in cluster_dict.keys():
                constraint = cluster_dict[key]
            else:
                continue

            scores, cls_pro= inference_recognizer(model, block_path)
            result = np.argmax(cls_pro * constraint)
            labels.append(label)
            preds.append(result)

            resultMatrix = classification_report(labels, preds, output_dict = True)

            print("%s\t%s\t%s\t%f\t%f\t%f\n" % (patch_name[0], label, result, resultMatrix['macro avg']['precision'], resultMatrix['macro avg']['recall'], resultMatrix['macro avg']['f1-score']))
            txt_matrix.write("%s\t%s\t%s\t%f\t%f\t%f\n" % (patch_name[0], label, result, resultMatrix['macro avg']['precision'], resultMatrix['macro avg']['recall'], resultMatrix['macro avg']['f1-score']))
        except:
            pass
        print("processing {}/{}".format(processing, total_sample))
        processing = processing + 1 

    txt_matrix.close()

    acc = accuracy_score(labels, preds)
    print('Acc: {}'.format(acc))
