# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
import numpy as np
import torch.utils.data
from data.base_dataset import BaseDataset
from data.augmentations import *
from data.DataProvider import DataProvider
# import data.cityscapes_dataset

def find_dataset_using_name(name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = name + '_loader'
    for _name, cls in datasetlib.__dict__.items():
        if _name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, logger):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt, logger)
    dataset = data_loader.load_data()
    return dataset

def get_composed_augmentations(opt):
    return Compose([RandomSized(opt.resize),
                    RandomCrop(opt.rcrop),
                    RandomHorizontallyFlip(opt.hflip)])

class CustomDatasetDataLoader():
    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger

        # status == 'train':
        source_data = find_dataset_using_name(opt.src_dataset)
        data_aug = None if opt.noaug else get_composed_augmentations(opt)
        self.source_train = source_data(opt, logger, augmentations=data_aug, split='all')
        logger.info("{} source dataset has been created".format(self.source_train.__class__.__name__))
        print("dataset {} for source was created".format(self.source_train.__class__.__name__))
        
        self.source_train_E = source_data(opt, logger, augmentations=data_aug, split='bai_easy')
        logger.info("{} source dataset has been created".format(self.source_train_E.__class__.__name__))
        print("dataset {} for source was created".format(self.source_train_E.__class__.__name__))

        self.source_train_H = source_data(opt, logger, augmentations=data_aug, split='bai_hard')
        logger.info("{} source dataset has been created".format(self.source_train_H.__class__.__name__))
        print("dataset {} for source was created".format(self.source_train_H.__class__.__name__))
        # status == 'train target':
        target_data = find_dataset_using_name(opt.tgt_dataset)
        self.target_train = target_data(opt, logger, augmentations=data_aug, split='train')
        logger.info("{} target dataset has been created".format(self.target_train.__class__.__name__))
        print("dataset {} for target was created".format(self.target_train.__class__.__name__))
   
        self.target_train_E = target_data(opt, logger, augmentations=data_aug, split='train_easy')
        logger.info("{} target dataset has been created".format(self.target_train_E.__class__.__name__))
        print("dataset {} for target was created".format(self.target_train_E.__class__.__name__))

        self.target_train_H = target_data(opt, logger, augmentations=data_aug, split='train_hard')
        logger.info("{} target dataset has been created".format(self.target_train_H.__class__.__name__))
        print("dataset {} for target was created".format(self.target_train_H.__class__.__name__))

        ## create train loader
        self.source_train_loader = DataProvider(
            dataset=self.source_train,
            batch_size=opt.bs,
            shuffle=not opt.noshuffle,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )

        self.source_train_loader_E = DataProvider(
            dataset=self.source_train_E,
            batch_size=opt.bs,
            shuffle=not opt.noshuffle,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )
        self.source_train_loader_H = DataProvider(
            dataset=self.source_train_H,
            batch_size=opt.bs,
            shuffle=not opt.noshuffle,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )
        self.target_train_loader_E = torch.utils.data.DataLoader(
            dataset=self.target_train_E,
            batch_size=opt.bs,
            shuffle=not opt.noshuffle,
            num_workers=1,
            drop_last=not opt.no_droplast,
            pin_memory=True,
        )
        self.target_train_loader_H = torch.utils.data.DataLoader(
            dataset=self.target_train_H,
            batch_size=opt.bs,
            shuffle=not opt.noshuffle,
            num_workers=1,
            drop_last=not opt.no_droplast,
            pin_memory=True,
        )
        self.target_train_loader = torch.utils.data.DataLoader(
            dataset=self.target_train,
            batch_size=opt.bs,
            shuffle=not opt.noshuffle,
            num_workers=1,
            drop_last=not opt.no_droplast,
            pin_memory=True,
        )

        # status == valid
        self.source_valid = None
        self.source_valid_loader = None

        self.target_valid = None
        self.target_valid_loader = None

        self.target_valid = target_data(opt, logger, augmentations=None, split='val')
        logger.info("{} target_valid dataset has been created".format(self.target_valid.__class__.__name__))
        print("dataset {} for target_valid was created".format(self.target_valid.__class__.__name__))

        self.target_valid_loader = torch.utils.data.DataLoader(
            self.target_valid,
            batch_size=opt.bs,
            shuffle=False,
            num_workers=int(opt.num_workers),
            drop_last=False,
            pin_memory=True,
        )
        # status == test

        self.target_test = target_data(opt, logger, augmentations=None, split='test')
        logger.info("{} target_valid dataset has been created".format(self.target_test.__class__.__name__))
        print("dataset {} for target_valid was created".format(self.target_test.__class__.__name__))

        self.target_test_loader = torch.utils.data.DataLoader(
            self.target_test,
            batch_size=opt.bs,
            shuffle=False,
            num_workers=int(opt.num_workers),
            drop_last=False,
            pin_memory=True,
        )

        #add
        
        # target_train_E_test = find_dataset_using_name(opt.tgt_dataset_E)
        # self.target_train_E_test = target_train_E_test(opt, logger, augmentations=None, split='train_easy')
        # self.target_train_E_test_loader = torch.utils.data.DataLoader(
        #     self.target_train_E_test,
        #     batch_size=opt.bs,
        #     shuffle=False,
        #     num_workers=int(opt.num_workers),
        #     drop_last=False,
        #     pin_memory=True,
        # )
        # target_train_H_test = find_dataset_using_name(opt.tgt_dataset_H)
        # self.target_train_H_test = target_train_H_test(opt, logger, augmentations=None, split='train_hard')
        # self.target_train_H_test_loader = torch.utils.data.DataLoader(
        #     self.target_train_H_test,
        #     batch_size=opt.bs,
        #     shuffle=False,
        #     num_workers=int(opt.num_workers),
        #     drop_last=False,
        #     pin_memory=True,
        # )

    def load_data(self):
        return self

