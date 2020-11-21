# -*- coding: utf-8 -*-
# @Time     : 2020/11/18 16:49
# @Author   : lishijie
import os
import argparse
import random
import numpy as np
from solvers import ObjectiveSolver

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(config):

    folder_path = {
        'ava': '/home/nlp/lsj/image_aesthetics_assessment/Database/AVA_dataset/',
    }

    img_num = {
        'ava': list(range(0, 204423))
    }

    sel_num = img_num[config.dataset]

    mseLoss_all = np.zeros(config.train_test_num, dtype=np.float)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        # Randomly select 80% images for training and the rest for testing
        random.shuffle(sel_num)
        train_index = sel_num[0:163539]
        test_index = sel_num[163539:204423]

        solver = ObjectiveSolver(config, folder_path[config.dataset], train_index, test_index, i)
        mseLoss_all[i] = solver.train()
    
    mseLoss_med = np.median(mseLoss_all)

    print('Testing median MSE %4.4f' % (mseLoss_med))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='ava', help='Support datasets: ava')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
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
    main(config)
