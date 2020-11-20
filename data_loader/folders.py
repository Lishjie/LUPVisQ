# -*- coding: utf-8 -*-
# @Time     : 2020/11/18 10:50
# @Author   : lishijie
from numpy.core.fromnumeric import var
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


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
