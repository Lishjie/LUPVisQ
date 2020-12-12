# -*- coding: utf-8 -*-
# @Time     : 2020/11/18 10:50
# @Author   : lishijie
from random import sample
from numpy.core.fromnumeric import var
from torch import le
import torch.utils.data as data
from PIL import Image, ImageFile
import numpy as np
import os
import os.path
import scipy.io
from openpyxl import load_workbook
import csv
ImageFile.LOAD_TRUNCATED_IMAGES = True


class AVAFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num, model_type):
        imgname = []
        label_all = []
        # ava_f = open(os.path.join(root, 'AVA_scores.txt'), 'r')
        ava_file = os.path.join(root, 'AVA_train_scores.csv')
        # ava_file = os.path.join(root, 'AVA_val_scores.csv')
        with open(ava_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['img_name'])
                label = row['avg_score'] if model_type == 'objective' else row['var_score']
                label = np.array(float(label)).astype(np.float32)
                label_all.append(label)
        
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'images', 'images', imgname[item]+'.jpg'), label_all[item]))
        
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


class SubjectiveNetDataset(data.Dataset):

    def __init__(self, root, index, transform, patch_num, database_type):
        imgname, label = getattr(self, database_type)(root)
        sample = []

        for i, item in enumerate(index):
            if i % 2 != 0:
                continue
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'images', 'images', imgname[item]+'.jpg'),
                                os.path.join(root, 'images', 'images', imgname[item+1]+'.jpg'),
                                1 if label[item] >= label[item+1] else -1))
        
        self.samples = sample
        self.transform = transform
    
    def __getitem__(self, index):
        path_main, path_sub, target = self.samples[index]
        sample_main = pil_loader(path_main)
        sample_main = self.transform(sample_main)
        sample_sub = pil_loader(path_sub)
        sample_sub = self.transform(sample_sub)
        return sample_main, sample_sub, target
    
    def __len__(self):
        length = len(self.samples)
        return length

    def ava_database(self, root):
        imgname = []
        label = []
        ava_file = os.path.join(root, 'AVA_subjective_train.csv')

        with open(ava_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_ = row['var_score']
                label_ = np.array(float(label_)).astype(np.float32)
                imgname.append(row['img_name'])
                label.append(label_)
        
        return imgname, label


class LUPVisQNetDataset(data.Dataset):

    def __init__(self, root, index, transform, patch_num, database_type):
        imgname, label_all_np = getattr(self, database_type)(root)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'images', 'images', imgname[item]+'.jpg'), label_all_np[item]))
        
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        length = len(self.samples)
        return length
    
    def ava_database(self, root):
        imgname = []
        label_all = []
        ava_file = os.path.join(root, 'AVA_trian_scores.csv')

        with open(ava_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['img_name'])
                label = []
                for i in range(1, 11):
                    label.append(row['score{}_num'.format(i)])
                label_all.append(label)
        label_all_np = np.array(label_all, dtype=np.float32)
        label_all_np = label_all_np.reshape(-1, 10, 1)

        return imgname, label_all_np


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
