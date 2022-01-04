# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import torch
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import matplotlib.image  as imgs
from PIL import Image
import random
import scipy.io as io
from tqdm import tqdm
from scipy import stats

from torch.utils import data

from data import BaseDataset
from data.randaugment import RandAugmentMC

class bai_loader(BaseDataset):
    """
    bai     dataset
    for domain adaptation to dian
    """

    colors = [  
        [0,0,0],
        [100,100,100],
    ]

    label_colours = dict(zip(range(2), colors))
    
    def __init__(self, opt, logger, augmentations=None,split='all'):
        self.opt = opt
        self.root = opt.src_rootpath
        self.split = split
        self.augmentations = augmentations
        self.randaug = RandAugmentMC(2, 10)
        self.n_classes = 2
        self.img_size = (512, 512)
        self.list_path='./Dataset/bai_list/{}.txt'.format(self.split)

        self.mean = [0.0, 0.0, 0.0] #TODO:  calculating the mean value of rgb channels on GTA5
        self.image_base_path = os.path.join(self.root, 'images')
        self.label_base_path = os.path.join(self.root, 'labels')

        with open(self.list_path) as f:
            self.ids = [i_id.strip() for i_id in f]

        self.valid_classes = [0,100]
        self.class_names = ["nomal","early-ECA",]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(2)))

        if len(self.ids) == 0:
            raise Exception(
                "No files for style=[%s] found in %s" % (self.split, self.image_base_path)
            )
        
        print("Found {} {} images".format(len(self.ids), self.split))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """__getitem__
        
        param: index
        """
        id = self.ids[index]
        img_path = os.path.join(self.image_base_path, id)
        lbl_path = os.path.join(self.label_base_path, id)

        
        img = Image.open(img_path)
        # print("open,{}".format(img_path))
        lbl = Image.open(lbl_path)

        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        img = np.asarray(img, dtype=np.uint8)
        lbl = np.asarray(lbl, dtype=np.uint8)

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        input_dict = {}        
        if self.augmentations!=None:
            img, lbl, _, _, _ = self.augmentations(img, lbl)
            img_strong, params = self.randaug(Image.fromarray(img))
            img_strong, _ = self.transform(img_strong, lbl)
            input_dict['img_strong'] = img_strong
            input_dict['params'] = params

        img, lbl = self.transform(img, lbl)

        input_dict['img'] = img
        input_dict['label'] = lbl
        input_dict['img_path'] = self.ids[index]
        return input_dict


    def encode_segmap(self, lbl):
        # for _i in self.void_classes:
        #     lbl[lbl == _i] = self.ignore_index
        # lbl[lbl == _i] = self.ignore_index
        lbl_copy = 250 * np.ones(lbl.shape, dtype=np.uint8)
        for _i in self.valid_classes:
            lbl_copy[lbl == _i] = self.class_map[_i]
        return lbl_copy

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def transform(self, img, lbl):
        """transform

        img, lbl
        """
        img = np.array(img)
        # img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, self.img_size, "nearest", mode='F')
        lbl = lbl.astype(int)
        
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes): 
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
