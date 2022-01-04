# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from parser_train import parser_, relative_path_to_absolute_path
os.environ["CUDA_VISIABLE_DIVICES"]="1"
from tqdm import tqdm
from data import create_dataset
from models import adaptation_modelv2
from utils import fliplr
from metrics import runningScore

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def cluster_subdomain(entropy_list, lambda1):
    easy=0.0
    hard=0.0
    num_e=0
    num_h=0

    entropy_list = sorted(entropy_list, key=lambda img: img[1])
    copy_list = entropy_list.copy()
    entropy_rank = [item[0] for item in entropy_list]

    easy_split = entropy_rank[ : int(len(entropy_rank) * lambda1)]
    hard_split = entropy_rank[int(len(entropy_rank)* lambda1): ]

    with open('./Dataset/dian_list/train_easy_proda.txt','w+') as f:
        for item in easy_split:
            f.write('%s\n' % item)

    with open('./Dataset/dian_list/train_hard_proda.txt','w+') as f:
        for item in hard_split:
            f.write('%s\n' % item)

    entropy_rank_value = [item for item in entropy_list]
    easy_value_split = entropy_rank_value[ : int(len(entropy_rank) * lambda1)]
    hard_value_split = entropy_rank_value[int(len(entropy_rank)* lambda1): ]

    # with open('./Dataset/dian_list/train_easy_proda_value.txt','w+') as f:
    #     for item in easy_value_split:
    #         f.write('%s\t%s\t%s\t%s\n' % (item[0],item[1],item[2],item[3]))
    #         easy+=item[2]
    #         num_e+=1

    # with open('./Dataset/dian_list/train_hard_proda_value.txt','w+') as f:
    #     for item in hard_value_split:
    #         f.write('%s\t%s\t%s\t%s\n' % (item[0],item[1],item[2],item[3]))
    #         hard+=item[2]
    #         num_h+=1
    # with open('./Dataset/dian_list/train_easy_proda_value.txt','r+') as f:
    #     f.write('tumor iou:%s\n' % str(easy/num_e))
    
    # with open('./Dataset/dian_list/train_hard_proda_value.txt','r+') as f:
    #     f.write('tumor iou:%s\n' % str(hard/num_h))
    return copy_list

def test(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(opt, logger) 

    if opt.model_name == 'deeplabv2':
        checkpoint = torch.load(opt.resume_path)['ResNet101']["model_state"]
        model = adaptation_modelv2.CustomModel(opt, logger,0)
        model.BaseNet.load_state_dict(checkpoint)
    
    running_metrics_val = runningScore(opt.n_class)
    validation(model, logger, datasets, device, opt,running_metrics_val)

def validation(model, logger, datasets, device, opt,running_metrics_val):
    _k = -1
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(datasets.target_train_loader, device, model, opt,running_metrics_val)
        #validate(datasets.target_valid_loader, device, model, opt)

def validate(valid_loader, device, model, opt,running_metrics_val):
    # ori_LP = os.path.join(opt.root, 'Code/ProDA', opt.save_path, opt.name)
    split_list=[]
    ori_LP = os.path.join(opt.root,  opt.save_path, opt.name)
    if not os.path.exists(ori_LP):
        os.makedirs(ori_LP)

    sm = torch.nn.Softmax(dim=1)
    for data_i in tqdm(valid_loader):
        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)
        filename = data_i['img_path']

        out = model.BaseNet_DP(images_val)
        outputUp = F.interpolate(out['out'], size=images_val.size()[2:], mode='bilinear', align_corners=True)
        
        #Miou
        pred = outputUp.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)
        score, class_iou = running_metrics_val.get_scores()
        running_metrics_val.reset()
        iou=class_iou[1]
            
        pred_trg_entropy = prob_2_entropy(F.softmax(outputUp))
        out_label= outputUp.cpu().data.numpy()
        # print("before",out_label.shape)
        out_label = out_label.transpose(1,2,0)
        out_label = np.argmax(out_label, axis=2)
        out_label=np.tile(out_label,(1,2,1,1))
        # print("after",out_label.shape)
        n= out_label.sum()/2
        if n<=100:
            entropy_mean = 1000
        else:
            pred_trg_entropy_all = (pred_trg_entropy.cpu()*out_label).sum().item()
            entropy_mean = (pred_trg_entropy_all/n)
            # entropy_mean = pred_trg_entropy.cpu().sum().item()/(images_val.size()[2]*images_val.size()[3])
        split_list.append((filename[0], entropy_mean, iou,n))
        

    cluster_subdomain(split_list, 0.7)
        
def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(logdir, 'run.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--save_path', type=str, default='./Pseudo', help='pseudo label update thred')
    parser.add_argument('--soft', action='store_true', help='save soft pseudo label')
    parser.add_argument('--flip', action='store_true')
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')
    opt.noaug = True
    opt.noshuffle = True

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    test(opt, logger)

#python generate_pseudo_label.py --name gta2citylabv2_warmup_soft --soft --resume_path ./logs/gta2citylabv2_warmup/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python generate_pseudo_label.py --name gta2citylabv2_stage1Denoise --flip --resume_path ./logs/gta2citylabv2_stage1Denoisev2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python generate_pseudo_label.py --name gta2citylabv2_stage2 --flip --resume_path ./logs/gta2citylabv2_stage2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast --bn_clr --student_init simclr
#python generate_pseudo_label.py --name syn2citylabv2_warmup_soft --soft --src_dataset synthia --n_class 16 --src_rootpath Dataset/SYNTHIA-RAND-CITYSCAPES --resume_path ./logs/syn2citylabv2_warmup/from_synthia_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast