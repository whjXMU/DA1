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
from tqdm import tqdm
from data import create_dataset
from utils import get_logger
from models import adaptation_modelv2
from utils.metrics import runningScore, averageMeter

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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

def structure(images,model1,model2,model_student):
    loss_kl=KL(images,model1,model_student)
    loss_kl+=KL(images,model2,model_student)
    print("loss_KL:{}".format(loss_kl))



def train(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    datasets = create_dataset(opt, logger)

    if opt.model_name == 'deeplabv2':
        model1 = adaptation_modelv2.CustomModel(opt, logger,0)
        model2 = adaptation_modelv2.CustomModel(opt, logger,0)
        model3 = adaptation_modelv2.CustomModel(opt, logger,0)
        model4 = adaptation_modelv2.CustomModel(opt, logger,0)
        model_student = adaptation_modelv2.CustomModel(opt, logger,0)
        checkpoint1 = torch.load("/home/hjw/proda-mine/logs/pro_warmup_KL_noL1_V2/from_bai_to_dian_on_deeplabv2_best_model1.pkl")['ResNet101']["model_state"]
        checkpoint2 = torch.load("/home/hjw/proda-mine/logs/pro_warmup_KL_noL1_V2/from_bai_to_dian_on_deeplabv2_best_model2.pkl")['ResNet101']["model_state"]
        checkpoint3 = torch.load("/home/hjw/proda-mine/logs/pro_warmup_KL_noL1_V2/from_bai_to_dian_on_deeplabv2_best_model3.pkl")['ResNet101']["model_state"]
        checkpoint4 = torch.load("/home/hjw/proda-mine/logs/pro_warmup_KL_noL1_V2/from_bai_to_dian_on_deeplabv2_best_model4.pkl")['ResNet101']["model_state"]
        model1.BaseNet.load_state_dict(checkpoint1)
        model1.BaseNet.load_state_dict(checkpoint2)
        model1.BaseNet.load_state_dict(checkpoint3)
        model1.BaseNet.load_state_dict(checkpoint4)
    # Setup Metrics
    running_metrics_val1 = runningScore(opt.n_class)
    time_meter = averageMeter()
    

    # load category anchors
    if opt.stage == 'stage1':
        objective_vectors = torch.load("/home/hjw/proda-mine/pretrained/bai2dianlabv2_warmup/prototypes_on_dian_from_deeplabv2")        
        model1.objective_vectors = torch.Tensor(objective_vectors).to(0)
        model2.objective_vectors = torch.Tensor(objective_vectors).to(0)
        model3.objective_vectors = torch.Tensor(objective_vectors).to(0)
        model4.objective_vectors = torch.Tensor(objective_vectors).to(0)
        model_student.objective_vectors = torch.Tensor(objective_vectors).to(0)
    # begin training
    # save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name))
    # save_path1 = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model1.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name))
    model1.iter = 0
    start_epoch = 0
    for epoch in range(start_epoch, opt.epochs):
        for data_i_E, data_i_H in zip(datasets.target_train_loader_E,datasets.target_train_loader_H):
            # print("len data_______________{}".format(len(data_i)))
        
            target_image_E = data_i_E['img'].to(device)
            target_imageS_E = data_i_E['img_strong'].to(device)
            target_params_E = data_i_E['params']
            target_image_full_E = data_i_E['img_full'].to(device)
            target_weak_params_E = data_i_E['weak_params']
            target_lp_E = data_i_E['lp'].to(device) if 'lp' in data_i_E.keys() else None
            target_lpsoft_E = data_i_E['lpsoft'].to(device) if 'lpsoft' in data_i_E.keys() else None

            target_image_H = data_i_H['img'].to(device)
            target_imageS_H = data_i_H['img_strong'].to(device)
            target_params_H = data_i_H['params']
            target_image_full_H = data_i_H['img_full'].to(device)
            target_weak_params_H = data_i_H['weak_params']
            target_lp_H = data_i_H['lp'].to(device) if 'lp' in data_i_H.keys() else None
            target_lpsoft_H = data_i_H['lpsoft'].to(device) if 'lpsoft' in data_i_H.keys() else None

            source_data = datasets.source_train_loader.next()


            model1.iter += 1
            i = model1.iter
            images_all = source_data['img'].to(device)
            labels_all = source_data['label'].to(device)



            start_ts = time.time()

            model1.eval(logger=logger)
            model2.eval(logger=logger)
            model3.eval(logger=logger)
            model4.eval(logger=logger)
            model_student.train(logger=logger)

            if opt.freeze_bn:
                # model1.freeze_bn_apply()
                # model2.freeze_bn_apply()
                # model3.freeze_bn_apply()
                # model4.freeze_bn_apply()
                model_student.freeze_bn_apply()
            # model1.optimizer_zerograd()
            # model2.optimizer_zerograd()
            # model3.optimizer_zerograd()
            # model4.optimizer_zerograd()
            model_student.optimizer_zerograd()


            # if opt.stage == 'warm_up':
                # loss_GTA, loss_G, loss_D = model1.step_adv(images, labels, target_image, source_imageS, source_params)
                # print("sucess 1 _______________")
            # elif opt.stage == 'stage1':
            if opt.stage == 'stage1':
                loss= model_student.seg(images_all, labels_all, target_image_E, target_imageS_E, target_params_E, target_lp_E,
                                        target_lpsoft_E, target_image_full_E, target_weak_params_E)
                structure(target_image_E,model1,model3,model_student)
                structure(target_image_H,model2,model4,model_student)

                
                
                # loss_KL1,loss_KL2=KL(model,model1,images, images1)
                # print("loss:{:.4f}, loss_CTS:{:.4f}, loss_consist:{:.4f} loss1:{:.4f}, loss_CTS1:{:.4f}, loss_consist1:{:.4f}, loss_KLL:{:.4f}".format(loss, loss_CTS, loss_consist,loss1, loss_CTS1, loss_consist1,loss_KL))

                
                
            time_meter.update(time.time() - start_ts)
            # print("sucess 2 _______________")
            #print(i)
            if (i + 1) % opt.print_interval == 0:
                if opt.stage == 'warm_up':
                    print('warm up')
                    # fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_GTA: {:.4f}  loss_G: {:.4f}  loss_D: {:.4f} Time/Image: {:.4f}"
                    # print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss_GTA, loss_G, loss_D, time_meter.avg / opt.bs)
                elif opt.stage == 'stage1':
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  Time/Image: {:.4f} \n loss: {:.4f}"
                    print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, time_meter.avg / opt.bs, loss)
                else:
                    print('stage 2')
                    # fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_GTA: {:.4f}  loss: {:.4f} Time/Image: {:.4f}"
                    # print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss_GTA, loss, time_meter.avg / opt.bs)
                print(print_str)
                logger.info(print_str)
                time_meter.reset()
            
            # evaluation
            if (i + 1) % opt.val_interval == 0:
                validation(model_student, logger, datasets, device, running_metrics_val1, iters = model1.iter, num="1",opt=opt)
                torch.cuda.empty_cache()
                logger.info('Model_student Best iou until now is {}'.format(model_student.best_iou))
                print('Model_student Best iou until now is {}'.format(model_student.best_iou))
            
            # print("sucess 3 _______________")
            # model1.scheduler_step()
            # model2.scheduler_step()
            # model3.scheduler_step()
            # model4.scheduler_step()
            model_student.scheduler_step()
        # print("sucess 4 _______________")

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
            #"optimizer_state": model.optimizers[_k].state_dict(),
            #"scheduler_state": model.schedulers[_k].state_dict(),  
            "objective_vectors": model.objective_vectors,
        }
        state[net.__class__.__name__] = new_state
    state['iter'] = iters + 1
    state['best_iou'] = class_iou[1]
    save_path = os.path.join(opt.logdir,"current_model_student.pkl")
    # torch.save(state, save_path)
    

    if class_iou[1] >= model.best_iou:
        torch.cuda.empty_cache()
        model.best_iou = class_iou[1]
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
        save_path = os.path.join(opt.logdir,"best_model_student.pkl")
        torch.save(state, save_path)
        return class_iou[1]

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
