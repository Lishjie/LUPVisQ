# -*- coding: utf-8 -*-
# @Time     : 2020/12/26 20:00
# @Author   : lishijie
from logging import Logger
import os
import argparse
import random
import numpy as np
import torch
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

from models.main_models import models
from data_loader import data_loader
from utils import setup_logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(config):

    folder_path = {
        'ava_database': '/home/nlp/lsj/image_aesthetics_assessment/Database/AVA_dataset/',
    }

    img_num = {
        'ava_database': list(range(0, 204423))
    }

    sel_num = img_num[config.dataset]
    # random.shuffle(sel_num)
    check_index = sel_num[0:100]
    check_loader = data_loader.LUPVisQDataLoader(config.dataset, folder_path[config.dataset], check_index, config.patch_size, config.check_patch_num, config.batch_size, config.num_workers, False)
    check_data = check_loader.get_data()

    logger.info('Check channel softmax on %s dataset for %d num samples' % (config.dataset, len(sel_num)))
    check(check_data, check_index)

def check(check_data, check_index):
    """Checking"""
    model_LUPVisQ = models.LUPVisQNet(14, 14, 14, class_num=config.class_num, backbone_type=config.backbone, channel_num=config.channel_num, tau=0.5).cuda()
    model_LUPVisQ.load_state_dict((torch.load('./result/ava_database/LUPVisQ/LUPVisQNet_ava_database_best_10_0.057.pth')))
    model_LUPVisQ.train(False)
    # gt_mean = []
    # gt_dis = []
    # channel_mean = []
    # channel_dis = []

    with torch.no_grad():
        for img, label, mean, std in check_data:
            img = torch.tensor(img.cuda())
            label = torch.tensor(label.cuda(), dtype=torch.float32)

            res = model_LUPVisQ(img, config.sample_num, istrain=True)

            # gt_mean += mean.float().cpu().tolist()
            # gt_dis += label.float().cpu().tolist()
            # channel_mean += cal_mean_std(res['score_distribution'].squeeze().cpu().tolist())
            # channel_dis += res['score_distribution'].squeeze().cpu().tolist()
            
            logger.info('gt_mean: {} \n channel_mean: {} \n gt_dis: {} \n channel_dis: {} \n\n'.format(mean.float().cpu().tolist(), cal_mean_std(res['score_distribution'].squeeze().cpu().tolist()), label.squeeze().float().cpu().tolist(), res['score_distribution'].squeeze().cpu().tolist()))

def cal_mean_std(score_dis):
    mean_batch = []

    for item in score_dis:
        mean = 0.0
        for score, num in enumerate(item, 1):
            mean += score * num
        mean_batch.append(mean)
    
    return mean_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ava_database', help='Support datasets: ava_database')
    parser.add_argument('--check_patch_num', dest='check_patch_num', type=int, default=1, help='Number of sample patches from checking image')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0, help='How many subprocesses are used to load data')
    parser.add_argument('--model_type', dest='model_type', type=str, default='LUPVisQ', help='objective | subjective | LUPVisQ')
    parser.add_argument('--sample_num', dest='sample_num', type=int, default=2, help='forward sample times')
    parser.add_argument('--backbone', dest='backbone', type=str, default='objectiveNet_backbone', help='backbone type')
    parser.add_argument('--class_num', dest='class_num', type=int, default=10, help='Number of scoring levels')
    parser.add_argument('--channel_num', dest='channel_num', type=int, default=5, help='Channel num of Multi-dimensional aesthetic channel')

    config = parser.parse_args()
    logger = setup_logger('./fix_channel_softmax.log')
    main(config)
