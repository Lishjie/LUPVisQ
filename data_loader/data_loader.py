# -*- coding: utf-8 -*-
# @Time     : 2020/11/16 20:28
# @Author   : lishijie
from random import shuffle
import torch
from torch.utils import data
from torch.utils.data import dataloader
import torchvision
from torchvision.transforms import transforms
from torchvision.transforms.transforms import Resize
from .folders import *


class DataLoader(object):
    """Dataset class for Aesthetic Visual Analysis database"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, num_workers=0, istrain=True, model_type='objective'):

        self.batch_size = batch_size
        self.istrain = istrain
        self.num_workers = num_workers
        self.model_type = model_type

        if dataset == 'ava':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        
        if dataset == 'ava':
            self.data = AVAFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num, model_type=self.model_type)
    
    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader


class SubjectiveDataLoader(object):
    """Dataset class for SubjectiveNet"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, num_workers=0, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain
        self.num_workers = num_workers

        if dataset == 'ava_database':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        
        if dataset == 'ava_database':
            self.data = SubjectiveNetDataset(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num, database_type=dataset)
    
    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader


class LUPVisQDataLoader(object):
    """Dataste class for LUPVisQNet"""

    def __init__(self, dataset, path, img_index, patch_size, patch_num, batch_size=1, num_workers=0, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain
        self.num_workers = num_workers

        if dataset == 'ava_database':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.299, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        
        if dataset == 'ava_database':
            self.data = LUPVisQNetDataset(
                root=path, index=img_index, transform=transforms, patch_num=patch_num, database_type=dataset)
        
    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, num_wokers=self.num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader

