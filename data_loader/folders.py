# -*- coding: utf-8 -*-
# @Time     : 2020/11/18 10:50
# @Author   : lishijie
from numpy.core.fromnumeric import var
import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
from openpyxl import load_workbook


class AVAFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num, model_type):
        imgname = []
        avg_all = []
        var_all = []
        ava_f = open(os.path.join(root, 'AVA_scores.txt'), 'r')
        lines = ava_f.readlines()
        for line in lines:
            item = line.strip().strip('\n').split(' ')
            imgname.append(item[1])
            avg_all.append(item[2])
            var_all.append(item[3])
        
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'images', 'images', imgname[item]+'.jpg'),
                               avg_all[item] if model_type == 'objectivity' else var_all[item]))
        
        self.samples = sample
        self.transform = transform
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target
    
    def __len__(self):
        length = len(self.samples)
        return length


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
