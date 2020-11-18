# -*- coding: utf-8 -*-
# @Time     : 2020/11/16 20:28
# @Author   : lishijie
import torch
from torch.utils import data
import torchvision
from torchvision.transforms.transforms import Resize
import folders

class Dataloader(object):
    """Dataset class for Aesthetic Visual Analysis database"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, num_workers=0, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain
        self.num_workers = num_workers

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
            self.data = folders.AVAFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
    
    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False, num_workers=0)
        return dataloader
