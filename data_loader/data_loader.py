# -*- coding: utf-8 -*-
# @Time     : 2021/01/02 16:26
# @Author   : lishijie
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.transforms import Resize
from .databases import *

class LUPVisQDataLoader(object):
    """Dataste class for LUPVisQNet"""
    def __init__(self, dataset, path, img_index, patch_size, sample_num, batch_size=1, num_workers=0, istrain='train'):
        self.dataset = dataset
        self.path = path
        self.img_index = img_index
        self.patch_size = patch_size
        self.sample_num = sample_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.istrain = istrain

        if dataset == 'ava_database':
            if istrain == 'train':
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((384, 384)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    # torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.Resize((299, 299)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.299, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((384, 384)),
                    torchvision.transforms.CenterCrop(size=patch_size),
                    # torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.Resize((299, 299)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        
        if dataset == 'ava_database':
            self.data = LUPVisQNetDataset(
                root=path, index=img_index, transform=transforms, sample_num=sample_num, database_type=dataset, istrain=istrain)
    
    def get_data(self):
        if self.istrain == 'train':
            dataloader = DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            dataloader = DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader
