# -*- coding: utf-8 -*-
# @Time     : 2020/12/02 10:48
# @Author   : lishijie
import torch
from scipy import stats
from pprint import pformat
import numpy as np
import models
import data_loader
import os
import time

from utils import setup_logger

class SubjectiveSolver(object):
    """Solver for training and testing SubjectiveNet"""
    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.train_patch_num = config.train_patch_num
        self.test_patch_num = config.test_patch_num
        self.dataset = config.dataset
        self.train_index = train_idx
        self.train_test_num = config.train_test_num
        self.batch_size = config.batch_size
        self.save_model_path = os.path.join(config.save_model_path, config.dataset)
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.logger = setup_logger(os.path.join(self.save_model_path, 'train_subjective.log'))
        self.logger_info(pformat(config))

        self.model_subjective = models.Subjective(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_subjective.train(True)

        self.mse_loss = torch.nn.MSELoss().cuda()

        backbone_params = list(map(id, self.model_subjective.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_subjective.parameters())
        self.lr = config.lr
        self.lrration = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrration},
                 {'params': self.model_subjective.res.parameters(), 'lr': self.lr},
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, num_workers=config.num_workers, istrain=True, model_type=config.model_type)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, batch_size=config.batch_size, num_workers=config.num_workers, istrain=False, model_type=config.model_type)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()
        self.logger_info('train with device {} and pytorch {}'.format(0, torch.__version__))
    
    def train(self):
        """Training"""
        best_srcc = 0.0  # Spearman’s rank order correlation coefficient
        best_plcc = 0.0  # Pearson’s linear correlation coefficient

        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            epoch_start = time.time()
            batch_num = 0

            for img, label in self.train_data:
                batch_num = batch_num + 1
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())
                batch_start = time.time()

                self.solver.zero_grad()

                # Generate weights for target network
                paras = self.model_subjective(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.mse_loss(pred.squeeze(), label.float().detach())
                if batch_num % 100 == 0:
                    batch_time = time.time() - batch_start
                    batch_start = time.time()
                    self.logger_info(
                        '[{}/{}], batch num: [{}/{}], MSE loss: {:.6f}, time: {:.2f}'.format(
                            t+1, self.epochs, batch_num, len(self.train_index)*self.train_patch_num // self.batch_size, loss, batch_time))
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                self.logger_info(
                    'Improved srcc from {} to {}'.format(test_plcc, best_srcc))
                best_srcc = test_srcc
                best_plcc = test_plcc
                trained_model_name = 'subjectiveNet_{}_best_{}.pth'.format(self.dataset, self.train_test_num)
                torch.save(self.model_subjective.state_dict(), os.path.join(self.save_model_path, trained_model_name))
                self.logger_info(
                    'Save model {} in path {}'.format(trained_model_name, self.save_model_path))
            epoch_time = time.time() - epoch_start
            self.logger_info(
                'Epoch: {}, Train_MSELoss: {}, Test_SRCC: {}, Test_LCC: {}, Best_SRCC: {}, time: {}'.format(
                    t + 1, sum(epoch_loss) / len(epoch_loss), test_srcc, test_plcc, best_srcc, epoch_time))
            # Update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_subjective.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)
        
        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc
    
    def test(self, data):
        """Testing"""
        self.model_subjective.train(False)
        pred_scores = []
        gt_scores = []

        for img, label in data:
            # Data.
            img = torch.tensor(img.cuda())
            label = torch.tensor(label.cuda())

            paras = self.model_subjective(img)
            model_target = models.TargetNet(paras).cuda()
            model_target.train(False)
            pred = model_target(paras['target_in_vec'])

            pred_scores.append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_subjective.train(True)
        return test_srcc, test_plcc

    def logger_info(self, s):
        self.logger.info(s)
