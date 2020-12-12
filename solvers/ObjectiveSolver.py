# -*- coding: utf-8 -*-
# @Time     : 2020/11/18 16:45
# @Author   : lishijie
import os
import time
from os import stat
import torch
from scipy import stats
from pprint import pformat
import numpy as np

from models.pretrain import models
from data_loader import data_loader
from utils import setup_logger

class ObjectiveSolver(object):
    def __init__(self, config, path, train_test_num=0, train_idx=[], test_idx=[]):

        self.epochs = config.epochs
        self.train_patch_num = config.train_patch_num
        self.test_patch_num = config.test_patch_num
        self.dataset = config.dataset
        self.train_index = train_idx
        self.train_test_num = train_test_num
        self.batch_size = config.batch_size
        self.save_model_path = os.path.join(config.save_model_path, config.dataset)
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.logger = setup_logger(os.path.join(self.save_model_path, 'train.log'))
        self.logger_info(pformat(config))

        self.model_objective = models.Objective(16, 224, 112, 56, 28, 14).cuda()
        self.model_objective.train(True)

        self.mse_loss = torch.nn.MSELoss().cuda()  # MSE loss

        backbone_params = list(map(id, self.model_objective.res.parameters()))  # get ResNet50 parameters
        self.fc_params = filter(lambda p: id(p) not in backbone_params, self.model_objective.parameters())  # get Fully Connected NetWork parameters
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [
                 {'params': self.fc_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_objective.res.parameters(), 'lr': self.lr},
                ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, num_workers=config.num_workers, istrain=True, model_type=config.model_type)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False, model_type=config.model_type)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()
        self.logger_info('train with device {} and pytorch {}'.format(0, torch.__version__))
    
    def train(self):
        """Training"""
        best_mseLoss = 100.0  # Mean Squared Error loss
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            epoch_start = time.time()
            bacth_start = time.time()
            batch_num = 0

            for img, label in self.train_data:
                batch_num = batch_num + 1
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())
                self.solver.zero_grad()

                # Quality prediction
                pred = self.model_objective(img)
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.mse_loss(pred.squeeze(), label.float().detach())
                if batch_num % 100 == 0:
                    batch_time = time.time() - bacth_start
                    bacth_start = time.time()
                    self.logger_info(
                        '[{}/{}], batch num: [{}/{}], loss: {:.6f}, time: {:.2f}'.format(
                            t+1, self.epochs, batch_num, len(self.train_index)*self.train_patch_num // self.batch_size, loss, batch_time))
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            test_mseLoss = self.test(self.test_data)
            if test_mseLoss < best_mseLoss:
                self.logger_info(
                    'Reduce MSE from {} to {}'.format(best_mseLoss, test_mseLoss))
                best_mseLoss = test_mseLoss
                trained_model_name = 'objectiveNet_{}_best_{}.pth'.format(self.dataset, self.train_test_num)
                torch.save(self.model_objective.state_dict(), os.path.join(self.save_model_path, trained_model_name))
                self.logger_info(
                    'Save model {} in path {}'.format(trained_model_name, self.save_model_path))
            epoch_time = time.time() - epoch_start
            self.logger_info(
                'Epoch: {}, Train_MSELoss: {}, Test_MSELoss: {}, time: {}'.format(
                    t + 1, sum(epoch_loss) / len(epoch_loss), test_mseLoss, epoch_time))
            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            paras = [
                     {'params': self.fc_params, 'lr': lr * self.lrratio},
                     {'params': self.model_objective.res.parameters(), 'lr': self.lr},
                    ]
            self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        print('Best test MSELoss %f' % (best_mseLoss))

        return best_mseLoss
    
    def test(self, data):
        """Testing"""
        self.model_objective.train(False)
        mse_scores = []

        for img, label in data:
            # Data
            img = torch.tensor(img.cuda())
            label = torch.tensor(label.cuda())

            pred = self.model_objective(img)

            mse = self.mse_loss(pred.squeeze(), label.float().detach())
            mse_scores.append(mse.item())
        
        self.model_objective.train(True)
        return sum(mse_scores) / len(mse_scores)

    def logger_info(self, s):
        self.logger.info(s)
