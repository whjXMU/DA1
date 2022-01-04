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

os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from tqdm import tqdm
from data import create_dataset
from utils import get_logger
from models import adaptation_model_focal
from utils.metrics import runningScore, averageMeter


def train(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    datasets = create_dataset(opt, logger)

    if opt.model_name == 'deeplabv2':
        model1 = adaptation_model_focal.CustomModel(opt, logger,0)
        model2 = adaptation_model_focal.CustomModel(opt, logger,0)
        model3 = adaptation_model_focal.CustomModel(opt, logger,0)
        model4 = adaptation_model_focal.CustomModel(opt, logger,0)
    # Setup Metrics
    running_metrics_val1 = runningScore(opt.n_class)
    running_metrics_val2 = runningScore(opt.n_class)
    running_metrics_val3 = runningScore(opt.n_class)
    running_metrics_val4 = runningScore(opt.n_class)
    time_meter = averageMeter()

    # load category anchors
    if opt.stage == 'stage1':
        objective_vectors = torch.load(os.path.join(os.path.dirname(opt.resume_path), 'prototypes_on_{}_from_{}'.format(opt.tgt_dataset, opt.model_name)))
        model1.objective_vectors = torch.Tensor(objective_vectors).to(0)
        model2.objective_vectors = torch.Tensor(objective_vectors).to(0)
        model3.objective_vectors = torch.Tensor(objective_vectors).to(0)
        model4.objective_vectors = torch.Tensor(objective_vectors).to(0)
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

            source_data_E = datasets.source_train_loader_E.next()
            source_data_H = datasets.source_train_loader_H.next()

            model1.iter += 1
            i = model1.iter
            images_E = source_data_E['img'].to(device)
            labels_E = source_data_E['label'].to(device)
            # source_imageS_E = source_data_E['img_strong'].to(device)
            # source_params_E = source_data_E['params']

            
            images_H = source_data_H['img'].to(device)
            labels_H = source_data_H['label'].to(device)
            # source_imageS_H = source_data_H['img_strong'].to(device)
            # source_params_H = source_data_H['params']


            start_ts = time.time()

            model1.train(logger=logger)
            model2.train(logger=logger)
            model3.train(logger=logger)
            model4.train(logger=logger)
            if opt.freeze_bn:
                model1.freeze_bn_apply()
                model2.freeze_bn_apply()
                model3.freeze_bn_apply()
                model4.freeze_bn_apply()
            model1.optimizer_zerograd()
            model2.optimizer_zerograd()
            model3.optimizer_zerograd()
            model4.optimizer_zerograd()

            # if opt.stage == 'warm_up':
                # loss_GTA, loss_G, loss_D = model1.step_adv(images, labels, target_image, source_imageS, source_params)
                # print("sucess 1 _______________")
            # elif opt.stage == 'stage1':
            if opt.stage == 'stage1':
                loss1, loss_CTS1, loss_consist1 = model1.step(images_E, labels_E, target_image_E, target_imageS_E, target_params_E, target_lp_E,
                                        target_lpsoft_E, target_image_full_E, target_weak_params_E)
                loss2, loss_CTS2, loss_consist2 = model2.step(images_E, labels_E, target_image_H, target_imageS_H, target_params_H, target_lp_H,
                                        target_lpsoft_H, target_image_full_H, target_weak_params_H)
                loss3, loss_CTS3, loss_consist3 = model3.step(images_H, labels_H, target_image_E, target_imageS_E, target_params_E, target_lp_E,
                                        target_lpsoft_E, target_image_full_E, target_weak_params_E)
                loss4, loss_CTS4, loss_consist4 = model4.step(images_H, labels_H, target_image_H, target_imageS_H, target_params_H, target_lp_H,
                                        target_lpsoft_H, target_image_full_H, target_weak_params_H)
                # structure(images_E,model1,model3,model4,model2)
                # structure(images_H,model3,model1,model2,model4)
                # structure(target_image_E,model1,model2,model4,model3)
                # structure(target_image_H,model2,model1,model3,model4)
                # loss_L1_H=L1_H(target_image_H,model1,model2,model3,model4)
                
                
                # loss_KL1,loss_KL2=KL(model,model1,images, images1)
                # print("loss:{:.4f}, loss_CTS:{:.4f}, loss_consist:{:.4f} loss1:{:.4f}, loss_CTS1:{:.4f}, loss_consist1:{:.4f}, loss_KLL:{:.4f}".format(loss, loss_CTS, loss_consist,loss1, loss_CTS1, loss_consist1,loss_KL))

                
                
            # else:
            #     loss_GTA, loss = model.step_distillation(images, labels, target_image, target_imageS, target_params, target_lp)

            time_meter.update(time.time() - start_ts)
            # print("sucess 2 _______________")
            #print(i)
            if (i + 1) % opt.print_interval == 0:
                if opt.stage == 'warm_up':
                    print('warm up')
                    # fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_GTA: {:.4f}  loss_G: {:.4f}  loss_D: {:.4f} Time/Image: {:.4f}"
                    # print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss_GTA, loss_G, loss_D, time_meter.avg / opt.bs)
                elif opt.stage == 'stage1':
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  Time/Image: {:.4f} \n loss1: {:.4f}  loss_CTS1: {:.4f}  loss_consist1: {:.4f} \nloss2: {:.4f}  loss_CTS2: {:.4f}  loss_consist2: {:.4f}\nloss3: {:.4f}  loss_CTS3: {:.4f}  loss_consist3: {:.4f}\nloss4: {:.4f}  loss_CTS4: {:.4f}  loss_consist4: {:.4f}"
                    print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, time_meter.avg / opt.bs,loss1, loss_CTS1, loss_consist1,loss2, loss_CTS2, loss_consist2,loss3, loss_CTS3, loss_consist3,loss4, loss_CTS4, loss_consist4)
                else:
                    print('stage 2')
                    # fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_GTA: {:.4f}  loss: {:.4f} Time/Image: {:.4f}"
                    # print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss_GTA, loss, time_meter.avg / opt.bs)
                print(print_str)
                logger.info(print_str)
                time_meter.reset()
            
            # evaluation
            # if (i + 1) % opt.val_interval == 0:
            if (i + 1) % 200 == 0:
                validation(model1, logger, datasets, device, running_metrics_val1, iters = model1.iter, num="1",opt=opt)
                validation(model2, logger, datasets, device, running_metrics_val2, iters = model1.iter, num="2",opt=opt)
                validation(model3, logger, datasets, device, running_metrics_val3, iters = model1.iter, num="3",opt=opt)
                validation(model4, logger, datasets, device, running_metrics_val4, iters = model1.iter, num="4",opt=opt)
                torch.cuda.empty_cache()
                logger.info('Model1 Best iou until now is {}'.format(model1.best_iou))
                logger.info('Model2 Best iou until now is {}'.format(model2.best_iou))
                logger.info('Model3 Best iou until now is {}'.format(model3.best_iou))
                logger.info('Model4 Best iou until now is {}'.format(model4.best_iou))
                print('Model1 Best iou until now is {}'.format(model1.best_iou))
                print('Model2 Best iou until now is {}'.format(model2.best_iou))
                print('Model3 Best iou until now is {}'.format(model3.best_iou))
                print('Model4 Best iou until now is {}'.format(model4.best_iou))
            
            # print("sucess 3 _______________")
            model1.scheduler_step()
            model2.scheduler_step()
            model3.scheduler_step()
            model4.scheduler_step()
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
    save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model{}.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name, num))
    # torch.save(state, save_path)
    

    if class_iou[1]>= model.best_iou:
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
        save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_best_model{}.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name,num))
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
