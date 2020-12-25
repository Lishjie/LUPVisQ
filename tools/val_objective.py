# -*- coding: utf-8 -*-
# @Time     : 2020/11/22 15:30
# @Author   : lishijie
from logging import Logger
import os
import argparse
import random
import numpy as np
from scipy import stats
import torch
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

from models.pretrain import models
from data_loader import data_loader
from utils import setup_logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(config):

    folder_path = {
        'ava': '/home/nlp/lsj/image_aesthetics_assessment/Database/AVA_dataset/',
    }

    img_num = {
        'ava': list(range(0, 51105))
        # 'ava': list(range(0, 1000))
    }

    sel_num = img_num[config.dataset]
    val_loader = data_loader.DataLoader(config.dataset, folder_path[config.dataset], sel_num, config.patch_size, config.val_patch_num, istrain=False, model_type=config.model_type, batch_size=config.batch_size, num_workers=config.num_workers)
    val_data = val_loader.get_data()

    logger.info('Validation on %s dataset for %d num samples' % (config.dataset, len(sel_num)))

    val_srcc, val_lcc, acc = val(val_data, sel_num)

    logger.info('val srcc: {}, val lcc: {}, acc: {}'.format(val_srcc, val_lcc, acc))

def val(val_data, val_idx):
    """Validation"""
    model_objective = models.Objective(16, 224, 112, 56, 28, 14).cuda()
    model_objective.train(False)
    model_objective.load_state_dict((torch.load('./result/ava_database/objective/objectiveNet_ava_best_0.pth')))
    pred_scores = []
    gt_scores = []
    acc_scores = []
    num = 0

    with torch.no_grad():
        for img, label in val_data:
            num = num + 1
            # Data
            img = torch.tensor(img.cuda())
            label = torch.tensor(label.cuda())

            pred = model_objective(img)

            pred_scores = pred_scores + pred.tolist()
            gt_scores = gt_scores + label.tolist()
            pred_np = np.array([0 if x <= 5.0 else 1 for x in pred.tolist()])
            gt_np = np.array([0 if x <= 5.0 else 1 for x in label.tolist()])
            acc_scores.append(sum(pred_np == gt_np) / pred.size(0))
            logger.info('iter num: [{}/{}]'.format(num, len(val_idx)*config.val_patch_num // config.batch_size))
    
    pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, config.val_patch_num)), axis=1)
    gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, config.val_patch_num)), axis=1)
    val_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    val_lcc, _ = stats.pearsonr(pred_scores, gt_scores)

    return val_srcc, val_lcc, sum(acc_scores) / len(acc_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ava', help='Support datasets: ava')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--val_patch_num', dest='val_patch_num', type=int, default=10, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')
    parser.add_argument('--save_model_path', dest='save_model_path', type=str, default='./result', help='Trained model save path')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0, help='How many subprocesses are used to load data')
    parser.add_argument('--model_type', dest='model_type', type=str, default='objective', help='objective | subjective | LUPVisQ')

    config = parser.parse_args()
    logger = setup_logger('./val.log')
    main(config)
