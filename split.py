# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from parser_train import parser_, relative_path_to_absolute_path
from tqdm import tqdm

from models.utils import freeze_bn, get_scheduler, cross_entropy2d

from data import create_dataset
from models import adaptation_modelv2
from metrics import runningScore
lambda_1=0.6

def cluster_subdomain(loss_list, lambda1):
    loss_list = sorted(loss_list, key=lambda img: img[1])
    copy_list = loss_list.copy()
    loss_rank = [item[0] for item in loss_list]

    easy_split = loss_rank[ : int(len(loss_rank) * lambda1)]
    hard_split = loss_rank[int(len(loss_rank)* lambda1): ]

    withw+open('./Dataset/bai_list/bai_easy_split.txt','w+') as f:
        for item in easy_split:
            f.write('%s\n' % item)

    with open('./Dataset/bai_list/bai_hard_split.txt','w+') as f:
        for item in hard_split:
            f.write('%s\n' % item)

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
        model = adaptation_modelv2.CustomModel(opt, logger)
        objective_vectors = torch.load(os.path.join(os.path.dirname(opt.resume_path), 'prototypes_on_{}_from_{}'.format(opt.tgt_dataset, opt.model_name)))
        model.objective_vectors = torch.Tensor(objective_vectors).to(0)
        # checkpoint = torch.load(opt.resume_path)['ResNet101']["model_state"]
        # model = adaptation_modelv2.CustomModel(opt, logger)
        # model.BaseNet.load_state_dict(checkpoint)
    
    running_metrics_val = runningScore(opt.n_class)

    validation(model, logger, datasets, device, running_metrics_val)

def validation(model, logger, datasets, device, running_metrics_val):
    _k = -1
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(datasets.source_train_loader, device, model, running_metrics_val)
        
    # score, class_iou = running_metrics_val.get_scores()
    # for k, v in score.items():
    #     print(k, v)
    #     logger.info('{}: {}'.format(k, v))

    # for k, v in class_iou.items():
    #     logger.info('{}: {}'.format(k, v))

    # running_metrics_val.reset()

    # torch.cuda.empty_cache()
    # return score["Mean IoU : \t"]

def validate(train_loader, device, model, running_metrics_val):
    loss_list=[]
    # train_loader_iter = enumerate(train_loader)
    sm = torch.nn.Softmax(dim=1)
    train_iter=0
    # print("lenth:{}".format(len(train_loader)))
    for data_i in tqdm(train_loader):
        train_iter+=1
        images_train = data_i['img'].to(device)
        labels_train = data_i['label'].to(device)

        outs = model.BaseNet_DP(images_train)
        #outputs = F.interpolate(sm(outs['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True)
        outputs = F.interpolate(outs['out'], size=images_train.size()[2:], mode='bilinear', align_corners=True)

        loss_GTA = cross_entropy2d(input=outputs, target=labels_train)
        loss_list.append((data_i['img_path'][0], loss_GTA.item() ))
        print("{},loss_GTA:{} ,".format(data_i['img_path'][0],loss_GTA))
        if train_iter >= 474:
            break
        # colorize_save(pred_trg_main, name[0])

    # split the enntropy_list into 
    cluster_subdomain(loss_list, lambda1=lambda_1)
        # pred = outputs.data.max(1)[1].cpu().numpy()
        # gt = labels_train.data.cpu().numpy()
        # running_metrics_val.update(gt, pred)

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
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    test(opt, logger)