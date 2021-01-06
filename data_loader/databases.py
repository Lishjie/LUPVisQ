# -*- coding: utf-8 -*-
# @Time     : 2021/01/02 10:07
# @Author   : lishijie
import os
import csv
import random
import numpy as np
from numpy.core.fromnumeric import mean, std
from numpy.core.records import array
from numpy.lib.function_base import append
import torch.utils.data as data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class LUPVisQNetDataset(data.Dataset):
    """
    Args:
        root: root path of dataset
        index: total number of dataset
        istrain: train | val | test
    """
    def __init__(self, root, index, transform, sample_num, database_type, istrain):
        self.root = root
        self.index = index
        self.transform = transform
        self.sample_num = sample_num
        self.database_type = database_type
        self.istrain = istrain
        imgname, score_all_np, score_weight_dict, mean_all, std_all = getattr(self, database_type)(root)

        sample = []
        for i, item in enumerate(index):
            count = 0
            if istrain == 'train':
                for key in score_weight_dict[item]:
                    if score_weight_dict[item][key] > 0:
                        count += 1
                        sample.append((os.path.join(root, 'images', 'images', imgname[item]+'.jpg'), np.array(float(key)).astype(np.float32),
                                        score_all_np[item], mean_all[item], std_all[item]))
                for aug in range(sample_num - count):
                    sample.append((os.path.join(root, 'images', 'images', imgname[item]+'.jpg'), random_weight_sample(score_weight_dict[item]),
                                    score_all_np[item], mean_all[item], std_all[item]))
            else:  # istrain: val or test
                sample.append((os.path.join(root, 'images', 'images', imgname[item]+'.jpg'), score_all_np[item], mean_all[item], std_all[item]))

        self.samples = sample
        self.transform = transform
    
    def __getitem__(self, index: int):
        if self.istrain == 'train':
            path, score, score_dis, mean_, std_ = self.samples[index]
            sample = pil_loader(path)
            sample = self.transform(sample)
            return sample, score
        else:  # val or test
            path, score_dis, mean_, std_ = self.samples[index]
            sample = pil_loader(path)
            sample = self.transform(sample)
            return sample, score_dis, mean_, std_
    
    def __len__(self):
        length = len(self.samples)
        return length

    def ava_database(self, root):
        imgname = []
        score_all = []
        score_weight_dict = []
        mean_all = []
        std_all = []
        ava_file = os.path.join(root, 'AVA_train_scores.csv' if self.istrain != 'test' else 'AVA_test_scores.csv')

        with open(ava_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['img_name'])
                score = []
                weight_dict = {}
                for i in range(1, 11):
                    score.append(row['score{}_num'.format(i)])
                    weight_dict[i] = int(row['score{}_num'.format(i)])
                score_all.append(score)
                score_weight_dict.append(weight_dict)
                mean_all.append(np.array(float(row['avg_score'])).astype(np.float32))
                std_all.append(np.array(float(row['var_score']) ** 0.5).astype(np.float32))
        score_all_np = np.array(score_all, dtype=np.float32)
        score_all_np = score_all_np / np.sum(score_all_np, axis=-1).reshape((score_all_np.shape[0], 1))
        score_all_np = score_all_np.reshape(-1, 10, 1)

        return imgname, score_all_np, score_weight_dict, mean_all, std_all


def random_weight_sample(weight_data):
    total = sum(weight_data.values())
    ra = random.uniform(0, total)
    curr_sum = 0
    ret = None
    keys = weight_data.keys()
    for k in keys:
        curr_sum += weight_data[k]
        if ra <= curr_sum:
            ret = k
            break
    return np.array(float(ret)).astype(np.float32)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
