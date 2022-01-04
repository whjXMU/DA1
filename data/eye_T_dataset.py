# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import numpy as np
import scipy.misc as m
from tqdm import tqdm

from torch.utils import data
from PIL import Image

from data.augmentations import *
from data.base_dataset import BaseDataset
from data.randaugment import RandAugmentMC

import random

# def recursive_glob(rootdir=".", suffix=""):
#     """Performs recursive glob with given suffix and rootdir 
#         :param rootdir is the root directory
#         :param suffix is the suffix to be searched
#     """
#     return [
#         os.path.join(looproot, filename)
#         for looproot, _, filenames in os.walk(rootdir)  #os.walk: traversal all files in rootdir and its subfolders
#         for filename in filenames
#         if filename.endswith(suffix)
#     ]

class eye_T_loader(BaseDataset):


    colors = [  # [  0,   0,   0],
        [0,0,0],
        [100,100,100],
        [100,100,0],
    ]

    label_colours = dict(zip(range(3), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "dian": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self, opt, logger, augmentations = None, split=''):
        """__init__

        :param opt: parameters of dataset
        :param writer: save the result of experiment
        :param logger: logging file
        :param augmentations: 
        """
        
        self.opt = opt
        self.root = opt.tgt_rootpath
        self.split = split
        self.augmentations = augmentations
        self.randaug = RandAugmentMC(2, 10)
        self.n_classes = opt.n_class
        self.img_size = (512,512)
        self.mean = np.array(self.mean_rgb['dian'])
        self.files = []
        self.paired_files = {}
        self.image_base_path = os.path.join(self.root, 'images')
        self.label_base_path = os.path.join(self.root, 'labels')
        self.list_path='./Dataset/eye_T_list/{}.txt'.format(self.split)
        # print('_______________________________________')
        # print(self.list_path)   
        with open(self.list_path) as f:
            # print(self.list_path)
            self.ids = [i_id.strip() for i_id in f]
            self.files =self.ids.copy()
        # print("file len")
        # print(len(self.files))
            # 
        
        # self.images_base = os.path.join(self.root, self.split)
        # self.annotations_base = os.path.join(
        #     self.root, self.split
        # )

        # self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".png")) #find all files from rootdir and subfolders with suffix = ".png"

        #self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        if self.n_classes == 3:
            self.valid_classes = [0,128,255]
            self.class_names = ["disc","cup","nomal"]
            
            self.to19 = dict(zip(range(3), range(3)))

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))   #zip: return tuples

        if not self.files:
            raise Exception(
                "No files for  split=[%s] found in %s" % (self.split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files), self.split))
    
    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        id = self.ids[index]
        img_path = os.path.join(self.image_base_path, id)
        lbl_path = os.path.join(self.label_base_path, id)
        

        img = Image.open(img_path)
        # print("open,{}".format(img_path))
        lbl = Image.open(lbl_path)
        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        img_full = img.copy().astype(np.float64)
        img_full -= self.mean
        img_full = img_full.astype(float) / 255.0
        img_full = img_full.transpose(2, 0, 1)

        lp, lpsoft, weak_params = None, None, None
        if self.split not in ['valid','test'] and self.opt.used_save_pseudo :
            if self.opt.proto_rectify:
                lpsoft = np.load(os.path.join(self.opt.path_soft, os.path.basename(img_path).replace('.png', '.npy')))
            else:
                lp_path = os.path.join(self.opt.path_LP, os.path.basename(img_path))
                lp = Image.open(lp_path)
                lp = lp.resize(self.img_size, Image.NEAREST)
                lp = np.array(lp, dtype=np.uint8)
                if self.opt.threshold:
                    conf = np.load(os.path.join(self.opt.path_LP, os.path.basename(img_path).replace('.png', '_conf.npy')))
                    lp[conf <= self.opt.threshold] = 250

        input_dict = {}
        if self.augmentations!=None:
            img, lbl, lp, lpsoft, weak_params = self.augmentations(img, lbl, lp, lpsoft,)
            img_strong, params = self.randaug(Image.fromarray(img))
            img_strong, _, _ = self.transform(img_strong, lbl)
            input_dict['img_strong'] = img_strong
            input_dict['params'] = params

        img, lbl_, lp = self.transform(img, lbl, lp)
                
        input_dict['img'] = img
        input_dict['img_full'] = torch.from_numpy(img_full).float()
        input_dict['label'] = lbl_
        input_dict['lp'] = lp
        input_dict['lpsoft'] = lpsoft
        input_dict['weak_params'] = weak_params  #full2weak
        input_dict['img_path'] = self.files[index]

        input_dict = {k:v for k, v in input_dict.items() if v is not None}
        return input_dict

    def transform(self, img, lbl, lp=None, check=True):
        """transform

        :param img:
        :param lbl:
        """
        # img = m.imresize(
        #     img, (self.img_size[0], self.img_size[1])
        # )  # uint8 with RGB mode
        img = np.array(img)
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if check and not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):   #todo: understanding the meaning 
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if lp is not None:
            classes = np.unique(lp)
            lp = np.array(lp)
            # if not np.all(np.unique(lp[lp != self.ignore_index]) < self.n_classes):
            #     raise ValueError("lp Segmentation map contained invalid class values")
        
            lp = torch.from_numpy(lp).long()

        return img, lbl, lp

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[self.to19[l]][0]
            g[temp == l] = self.label_colours[self.to19[l]][1]
            b[temp == l] = self.label_colours[self.to19[l]][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        label_copy = 250 * np.ones(mask.shape, dtype=np.uint8)
        for k, v in list(self.class_map.items()):
            label_copy[mask == k] = v
        return label_copy

    def get_cls_num_list(self):
        cls_num_list = np.array([1557726944,  254364912,  673500400,   18431664,   14431392,
                                29361440,    7038112,    7352368,  477239920,   40134240,
                                211669120,   36057968,     865184,  264786464,   17128544,
                                2385680,     943312,     504112,    2174560])
        return cls_num_list

# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT License.

# import os
# import sys
# import torch
# import numpy as np
# import scipy.misc as m
# import matplotlib.pyplot as plt
# import matplotlib.image  as imgs
# from PIL import Image
# import random
# import scipy.io as io
# from tqdm import tqdm
# from scipy import stats

# from torch.utils import data

# from data import BaseDataset
# from data.randaugment import RandAugmentMC

# class eye_T_loader(BaseDataset):
#     """
#     bai     dataset
#     for domain adaptation to dian
#     """

#     colors = [  
#         [0,0,0],
#         [100,100,100],
#         [100,100,0],
#     ]

#     label_colours = dict(zip(range(3), colors))
#     def __init__(self, opt, logger, augmentations=None):
#         self.opt = opt
#         self.root = opt.src_rootpath
#         self.split = 'val'
#         self.augmentations = augmentations
#         self.randaug = RandAugmentMC(2, 10)
#         self.n_classes = 3
#         self.img_size = (512, 512)
#         self.list_path='./Dataset/eye_list/{}.txt'.format(self.split)

#         self.mean = [0.0, 0.0, 0.0] #TODO:  calculating the mean value of rgb channels on GTA5
#         self.image_base_path = os.path.join(self.root, 'images')
#         self.label_base_path = os.path.join(self.root, 'labels')

#         with open(self.list_path) as f:
#             self.ids = [i_id.strip() for i_id in f]
        
#         # self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
#         self.valid_classes = [0,128,255]
#         self.class_names = ["disc","cup","nomal"]

#         self.ignore_index = 250
#         self.class_map = dict(zip(self.valid_classes, range(3)))

#         if len(self.ids) == 0:
#             raise Exception(
#                 "No files for style=[%s] found in %s" % (self.split, self.image_base_path)
#             )
        
#         print("Found {} {} images".format(len(self.ids), self.split))

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, index):
#         """__getitem__
        
#         param: index
#         """
#         id = self.ids[index]
#         img_path = os.path.join(self.image_base_path, id).replace('jpg','png')
#         # lbl_path = os.path.join(self.label_base_path, id)
#         lbl_path=img_path.replace('images','labels').replace('jpg','png')
        
#         img = Image.open(img_path)
#         # print("open,{}".format(img_path))
#         lbl = Image.open(lbl_path)

#         img = img.resize(self.img_size, Image.BILINEAR)
#         lbl = lbl.resize(self.img_size, Image.NEAREST)
#         img = np.asarray(img, dtype=np.uint8)
#         lbl = np.asarray(lbl, dtype=np.uint8)

#         lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

#         input_dict = {}        
#         if self.augmentations!=None:
#             img, lbl, _, _, _ = self.augmentations(img, lbl)
#             img_strong, params = self.randaug(Image.fromarray(img))
#             img_strong, _ = self.transform(img_strong, lbl)
#             input_dict['img_strong'] = img_strong
#             input_dict['params'] = params

#         img, lbl = self.transform(img, lbl)

#         input_dict['img'] = img
#         input_dict['label'] = lbl
#         input_dict['img_path'] = self.ids[index]
#         return input_dict


#     def encode_segmap(self, lbl):
#         # for _i in self.void_classes:
#         #     lbl[lbl == _i] = self.ignore_index
#         # lbl[lbl == _i] = self.ignore_index
#         lbl_copy = 250 * np.ones(lbl.shape, dtype=np.uint8)
#         for _i in self.valid_classes:
#             lbl_copy[lbl == _i] = self.class_map[_i]
#         return lbl_copy

#     def decode_segmap(self, temp):
#         r = temp.copy()
#         g = temp.copy()
#         b = temp.copy()
#         for l in range(0, self.n_classes):
#             r[temp == l] = self.label_colours[l][0]
#             g[temp == l] = self.label_colours[l][1]
#             b[temp == l] = self.label_colours[l][2]

#         rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
#         rgb[:, :, 0] = r / 255.0
#         rgb[:, :, 1] = g / 255.0
#         rgb[:, :, 2] = b / 255.0
#         return rgb

#     def transform(self, img, lbl):
#         """transform

#         img, lbl
#         """
#         img = np.array(img)
#         # img = img[:, :, ::-1] # RGB -> BGR
#         img = img.astype(np.float64)
#         img -= self.mean
#         img = img.astype(float) / 255.0
#         img = img.transpose(2, 0, 1)

#         classes = np.unique(lbl)
#         lbl = np.array(lbl)
#         lbl = lbl.astype(float)
#         # lbl = m.imresize(lbl, self.img_size, "nearest", mode='F')
#         lbl = lbl.astype(int)
        
#         if not np.all(classes == np.unique(lbl)):
#             print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

#         if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes): 
#             print("after det", classes, np.unique(lbl))
#             raise ValueError("Segmentation map contained invalid class values")
        
#         img = torch.from_numpy(img).float()
#         lbl = torch.from_numpy(lbl).long()

#         return img, lbl
