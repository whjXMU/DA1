# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
sys.path.append("..")
sys.path.append("../utils")
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch
from torch.cuda.memory import list_gpu_processes
import torch.nn.functional as F
from torch.nn.modules.loss import L1Loss
from utils.parser_train import parser_, relative_path_to_absolute_path

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from tqdm import tqdm
from data import create_dataset
from utils import get_logger
from models import adaptation_modelv2
from utils.metrics import runningScore, averageMeter

def KL(images,model1,model2):

    source_out1 = model1.BaseNet_DP(images, ssl=True)
    source_outputUp1 = F.interpolate(source_out1['out'].detach(), size=images.size()[2:], mode='bilinear', align_corners=True)
    source_outputUp1 = F.softmax(source_outputUp1, dim=-1)
    # print(" shape{}".format(source_outputUp.shape))
    source_out2 = model2.BaseNet_DP(images, ssl=True)
    source_outputUp2 = F.interpolate(source_out2['out'], size=images.size()[2:], mode='bilinear', align_corners=True)
    source_outputUp2 = F.log_softmax(source_outputUp2 , dim=-1)
    loss_kl = F.kl_div(source_outputUp2, source_outputUp1, reduction='none')
    # print(loss_1.sum())
    # print("loss_1_mean:{},loss_1a:{}".format(loss_1.mean().item()*3000,loss_1a.sum().item()))
    loss_kl=0.01*loss_kl.sum()
    loss_kl.backward()
    model2.BaseOpti.step()
    model2.BaseOpti.zero_grad()
    if hasattr(torch.cuda,"empty_cache"):
        torch.cuda.empty_cache()
    # print("loss_1:{} ".format(loss_1))
    return float(loss_kl)

def L1_H(images,model1,model2,model3,model4):
    loss_L1=L1Loss()
    
    source_out1 = model1.BaseNet_DP(images, ssl=True)
    source_outputUp1 = F.interpolate(source_out1['out'], size=images.size()[2:], mode='bilinear', align_corners=True)
    source_outputUp1 = F.softmax(source_outputUp1, dim=-1)
    # print(" shape{}".format(source_outputUp.shape))
    source_out2 = model2.BaseNet_DP(images, ssl=True)
    source_outputUp2 = F.interpolate(source_out2['out'], size=images.size()[2:], mode='bilinear', align_corners=True)
    source_outputUp2 = F.softmax(source_outputUp2 , dim=-1)
    source_out3 = model3.BaseNet_DP(images, ssl=True)
    source_outputUp3 = F.interpolate(source_out3['out'], size=images.size()[2:], mode='bilinear', align_corners=True)
    source_outputUp3 = F.softmax(source_outputUp3 , dim=-1)
    source_out4 = model4.BaseNet_DP(images, ssl=True)
    source_outputUp4 = F.interpolate(source_out4['out'], size=images.size()[2:], mode='bilinear', align_corners=True)
    source_outputUp4 = F.softmax(source_outputUp4, dim=-1)
    loss_l1=1000*loss_L1(source_outputUp1,source_outputUp2)+loss_L1(source_outputUp3,source_outputUp4)+loss_L1(source_outputUp2,source_outputUp4)
    print("loss_l1:{}".format(loss_l1))
    loss_l1.backward()
    model1.BaseOpti.step()
    model1.BaseOpti.zero_grad()
    model2.BaseOpti.step()
    model2.BaseOpti.zero_grad()
    model3.BaseOpti.step()
    model3.BaseOpti.zero_grad()
    model4.BaseOpti.step()
    model4.BaseOpti.zero_grad()
    return loss_l1

def L1(images,model1,model2):
    loss_L1=L1Loss()
    
    source_out1 = model1.BaseNet_DP(images, ssl=True)
    source_outputUp1 = F.interpolate(source_out1['out'], size=images.size()[2:], mode='bilinear', align_corners=True)
    source_outputUp1 = F.softmax(source_outputUp1, dim=-1)
    # print(" shape{}".format(source_outputUp.shape))
    source_out2 = model2.BaseNet_DP(images, ssl=True)
    source_outputUp2 = F.interpolate(source_out2['out'], size=images.size()[2:], mode='bilinear', align_corners=True)
    source_outputUp2 = F.softmax(source_outputUp2 , dim=-1)
    loss_l1=loss_L1(source_outputUp1,source_outputUp2)*1000
    loss_l1.backward()
    model1.BaseOpti.step()
    model1.BaseOpti.zero_grad()
    model2.BaseOpti.step()
    model2.BaseOpti.zero_grad()
    return loss_l1

def structure(images,model1,model2,model3,model4):
    loss_kl=KL(images,model1,model2)
    loss_kl+=KL(images,model1,model3)
    loss_L1=L1(images,model1,model4)
    print("loss_KL:{}, loss_L1:{}".format(loss_kl,loss_L1))



def train(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    num=0
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    datasets = create_dataset(opt, logger)

    if opt.model_name == 'deeplabv2':
        model1 = adaptation_modelv2.CustomModel(opt, logger,0)
        checkpoint1 = torch.load("/home/hjw/proda-mine/logs/proda_warmup_data0.7_easy/S_all_T_easy_best.pkl")['ResNet101']["model_state"]
        model1.BaseNet.load_state_dict(checkpoint1)
    # Setup Metrics
    running_metrics_val1 = runningScore(opt.n_class)

    time_meter = averageMeter()

    # load category anchors
    if opt.stage == 'stage1':
        
        objective_vectors = torch.load("/home/hjw/proda-mine/pretrained/bai2dianlabv2_warmup/prototypes_on_dian_from_deeplabv2")
        # objective_vectors = torch.load(os.path.join(os.path.dirname(opt.resume_path), 'prototypes_on_{}_from_{}'.format(opt.tgt_dataset, opt.model_name)))
        model1.objective_vectors = torch.Tensor(objective_vectors).to(0)
    # begin training
    # save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name))
    # save_path1 = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model1.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name))
    model1.iter = 0
    start_epoch = 0
    for epoch in range(start_epoch, opt.epochs):
        for data_i_E_label, data_i_H in zip(datasets.target_train_loader_E_label,datasets.target_train_loader_H):
            # print("len data_______________{}".format(len(data_i)))
            
            # target_image_E = data_i_E['img'].to(device)
            # target_imageS_E = data_i_E['img_strong'].to(device)
            # target_params_E = data_i_E['params']
            # target_image_full_E = data_i_E['img_full'].to(device)
            # target_weak_params_E = data_i_E['weak_params']
            # target_lp_E = data_i_E['lp'].to(device) if 'lp' in data_i_E.keys() else None
            # target_lpsoft_E = data_i_E['lpsoft'].to(device) if 'lpsoft' in data_i_E.keys() else None

            target_image_H = data_i_H['img'].to(device)
            target_imageS_H = data_i_H['img_strong'].to(device)
            target_params_H = data_i_H['params']
            target_image_full_H = data_i_H['img_full'].to(device)
            target_weak_params_H = data_i_H['weak_params']
            target_lp_H = data_i_H['lp'].to(device) if 'lp' in data_i_H.keys() else None
            target_lpsoft_H = data_i_H['lpsoft'].to(device) if 'lpsoft' in data_i_H.keys() else None

            source_data = datasets.source_train_loader.next()
            # source_data_H = datasets.source_train_loader_H.next()

            model1.iter += 1
            i = model1.iter
            images = source_data['img'].to(device)
            labels = source_data['label'].to(device)
            # source_imageS_E = source_data_E['img_strong'].to(device)
            # source_params_E = source_data_E['params']

            
            # images_H = source_data_H['img'].to(device)
            # labels_H = source_data_H['label'].to(device)
            target_E_image=data_i_E_label['img'].to(device)
            target_E_label=data_i_E_label['label'].to(device)
            # source_imageS_H = source_data_H['img_strong'].to(device)
            # source_params_H = source_data_H['params']


            start_ts = time.time()

            model1.train(logger=logger)

            if opt.freeze_bn:
                model1.freeze_bn_apply()

            model1.optimizer_zerograd()

            if opt.stage == 'stage1':
                if num%2==0:
                    # print('train_source',num)
                    loss1, loss_CTS1, loss_consist1 = model1.step(images, labels, target_image_H, target_imageS_H, target_params_H, target_lp_H,
                                        target_lpsoft_H, target_image_full_H, target_weak_params_H)
                    num=num+1
                else :
                    # print('train_target_easy')
                    loss1, loss_CTS1, loss_consist1 = model1.step(target_E_image, target_E_label, target_image_H, target_imageS_H, target_params_H, target_lp_H,
                                        target_lpsoft_H, target_image_full_H, target_weak_params_H)
                    num=num+1
                
            # else:
            #     loss_GTA, loss = model.step_distillation(images, labels, target_image, target_imageS, target_params, target_lp)

            time_meter.update(time.time() - start_ts)
            # print("sucess 2 _______________")
            #print(i)
            if (i + 1) % opt.print_interval == 0:
                if opt.stage == 'warm_up':
                    print('warm up')
                elif opt.stage == 'stage1':
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  Time/Image: {:.4f} \n loss1: {:.4f}  loss_CTS1: {:.4f}  loss_consist1: {:.4f} "
                    print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, time_meter.avg / opt.bs,loss1, loss_CTS1, loss_consist1)
                else:
                    print('stage 2')
                    # fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_GTA: {:.4f}  loss: {:.4f} Time/Image: {:.4f}"
                    # print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss_GTA, loss, time_meter.avg / opt.bs)
                print(print_str)
                logger.info(print_str)
                time_meter.reset()
            
            # evaluation
            if (i + 1) % opt.val_interval == 0:
                validation(model1, logger, datasets, device, running_metrics_val1, iters = model1.iter, num="1",opt=opt)

                torch.cuda.empty_cache()
                logger.info('Model1 Best iou until now is {}'.format(model1.best_iou))
                print('Model1 Best iou until now is {}'.format(model1.best_iou))


            model1.scheduler_step()


def validation(model, logger, datasets, device, running_metrics_val, iters,num, opt=None):
    iters = iters
    _k = -1
    for v in model.optimizers:
        _k += 1
        for param_group in v.param_groups:
            _learning_rate = param_group.get('lr')
        logger.info("learning rate is {} for {} net".format(_learning_rate, model.nets[_k].__class__.__name__))
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    val_datset = datasets.target_valid_loader
    #val_datset = datasets.target_train_loader
    with torch.no_grad():
        validate(val_datset, device, model, running_metrics_val)

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))

    for k, v in class_iou.items():
        logger.info('{}: {}'.format(k, v))

    running_metrics_val.reset()

    torch.cuda.empty_cache()
    state = {}
    _k = -1
    for net in model.nets:
        _k += 1
        new_state = {
            "model_state": net.state_dict(),
            "objective_vectors": model.objective_vectors,
        }
        state[net.__class__.__name__] = new_state
    state['iter'] = iters + 1
    state['best_iou'] = score["Mean IoU : \t"]
    save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model{}_hard.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name, num))

    if score["Mean IoU : \t"] >= model.best_iou:
        torch.cuda.empty_cache()
        model.best_iou = score["Mean IoU : \t"]
        state = {}
        _k = -1
        for net in model.nets:
            _k += 1
            new_state = {
                "model_state": net.state_dict(),
                #"optimizer_state": model.optimizers[_k].state_dict(),
                #"scheduler_state": model.schedulers[_k].state_dict(),     
                "objective_vectors": model.objective_vectors,                
            }
            state[net.__class__.__name__] = new_state
        state['iter'] = iters + 1
        state['best_iou'] = model.best_iou
        save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_best_model{}_hard.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name,"E2H"))
        torch.save(state, save_path)
        return score["Mean IoU : \t"]

def validate(valid_loader, device, model, running_metrics_val):
    for data_i in tqdm(valid_loader):

        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)

        out = model.BaseNet_DP(images_val)

        outputs = F.interpolate(out['out'], size=images_val.size()[2:], mode='bilinear', align_corners=True)
        #val_loss = loss_fn(input=outputs, target=labels_val)

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    train(opt, logger)
