# -*- coding: utf-8 -*-
# @Time     : 2021/01/04 22:32
# @Author   : lishijie
import os
import argparse
import random
import numpy as np
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

from solvers import LUPVisQSolver

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(config):
    
    folder_path = {
        'ava_database': '/home/nlp/lsj/image_aesthetics_assessment/Database/AVA_dataset/',
    }

    img_num = {
        'ava_database': list(range(0, 204423))
        # 'ava_database': list(range(0, 5000))
    }

    sel_num = img_num[config.dataset]

    edmLoss_all = np.zeros(config.train_test_num, dtype=np.float)

    print('Training and validation on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        # Randomly select 80% images for training and the rest for validation
        random.shuffle(sel_num)
        train_index = sel_num[0:163539]
        val_index = sel_num[163539:204423]
        # train_index = sel_num[0:50]
        # val_index = sel_num[50:100]

        solver = LUPVisQSolver(config, folder_path[config.dataset], train_index, val_index)
        edmLoss_all[i] = solver.train()
    
    emdLoss_med = np.median(edmLoss_all)

    print('Validation median EMD %4.4f' % (emdLoss_med))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base config
    parser.add_argument('--dataset', dest='dataset', type=str, default='ava_database', help='Support datasets: ava')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--train_sample_num', dest='train_sample_num', type=int, default=15, help='Number of sample patches from training image')
    parser.add_argument('--val_sample_num', dest='val_sample_num', type=int, default=1, help='Number of sample patches from validation image')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=336, help='Crop size for training & validation image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-val times')
    parser.add_argument('--save_model_path', dest='save_model_path', type=str, default='./result', help='Trained model save path')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0, help='How many subprocesses are used to load data')
    # model config
    parser.add_argument('--model_type', dest='model_type', type=str, default='LUPVisQ', help='objective | subjective | LUPVisQ')
    parser.add_argument('--repeat_num', dest='repeat_num', type=int, default=1000, help='forward sample times')
    parser.add_argument('--class_num', dest='class_num', type=int, default=10, help='Number of scoring levels')
    parser.add_argument('--channel_num', dest='channel_num', type=int, default=5, help='Channel num of Multi-dimensional aesthetic channel')
    parser.add_argument('--backbone_type', dest='backbone_type', type=str, default='inceptionv3_torchmodel', help='inceptionv3_torchmodel | resnet50_torchmodel')
    parser.add_argument('--tau', dest='tau', type=float, default=0.1, help='temperature for Gunbel-Softmax')
    # optimizer
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lr', dest='lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--lr_decay_rate', dest='lr_decay_rate', type=float, default=0.9)
    parser.add_argument('--lambda_', dest='lambda_', type=float, default=1e-3)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    # parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    # parser.add_argument('--lr_decay_rate', dest='lr_decay_rate', type=float, default=0.95, help='Learning rate decay rate')
    # parser.add_argument('--lr_decay_freq', dest='lr_decay_freq', type=float, default=10, help="Learning rate decay frequency")
    # parser.add_argument('--backbone', dest='backbone', type=str, default='objectiveNet_backbone', help='backbone type')

    config = parser.parse_args()
    main(config) 
