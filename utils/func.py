import numpy as np
import torch
import torch.nn as nn
import logging
import os
import datetime
from torch.nn.modules.loss import L1Loss
import torch.nn.functional as F

from loss import cross_entropy_2d


def dice(binary_segmentation, binary_gt_label):
    """
    Compute the Dice coefficient between two binary segmentation.
    Dice coefficient is defined as here:
    https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Input:
        binary_segmentation: binary 2D numpy array representing the region of
        interest as segmented by the algorithm
        binary_gt_label: binary 2D numpy array representing the region of
        interest as provided in the database
    Output:
        dice_value: Dice coefficient between the segmentation and the ground
        truth
    """ 

    # turn all variables to booleans, just in case
    binary_segmentation = np.asarray(binary_segmentation, dtype=np.bool)
    binary_gt_label = np.asarray(binary_gt_label, dtype=np.bool)

    # compute the intersection
    intersection = np.logical_and(binary_segmentation, binary_gt_label)

    # count the number of True pixels in the binary segmentation
    segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
    # same for the ground truth
    gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
    # same for the intersection
    intersection = float(np.sum(intersection.flatten()))

    # compute the Dice coefficient
    dice_value = 2 * intersection / (segmentation_pixels + gt_label_pixels)

    return dice_value

def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().to(device)
    return cross_entropy_2d(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, cfg, learning_rate):
    lr = lr_poly(learning_rate, i_iter, cfg.TRAIN.MAX_ITERS, cfg.TRAIN.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate(optimizer, i_iter, cfg):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE)


def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, cfg, cfg.TRAIN.LEARNING_RATE_D)


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def KL(images,model1,model2):
    
    source_out1 = model1.BaseNet_DP(images)
    teacher = F.interpolate(source_out1['out'].detach(), size=images.size()[2:], mode='bilinear', align_corners=True)
    teacher = F.softmax(teacher, dim=1)
    # print(" shape{}".format(source_outputUp.shape))
    source_out2 = model2.BaseNet_DP(images, ssl=True)
    student = F.interpolate(source_out2['out'], size=images.size()[2:], mode='bilinear', align_corners=True)
    # source_outputUp2 = F.log_softmax(source_outputUp2 , dim=-1)
    student = F.log_softmax(student , dim=-1)
    loss_kl = 0.01*F.kl_div(student, teacher, reduction='none')
    # print(loss_1.sum())
    # print("loss_1_mean:{},loss_1a:{}".format(loss_1.mean().item()*3000,loss_1a.sum().item()))
    # loss_kl=0.01*loss_kl.sum()
    mask = (teacher != 250).float()
    loss_kl = (loss_kl * mask).sum() / mask.sum()
    loss_kl.backward()
    model2.BaseOpti.step()
    model2.BaseOpti.zero_grad()
    if hasattr(torch.cuda,"empty_cache"):
        torch.cuda.empty_cache()
    # print("loss_1:{} ".format(loss_1))
    return float(loss_kl)


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
    # loss_L1=L1(images,model1,model4)
    # loss_L1+=L1(images,model2,model3)
    # print("loss_KL:{}, loss_L1:{}".format(loss_kl,loss_L1))
    print("loss_KL:{}".format(loss_kl))
    # logger.info("loss_KL:{}".format(loss_kl))