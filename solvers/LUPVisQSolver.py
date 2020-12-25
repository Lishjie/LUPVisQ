# -*- coding: utf-8 -*-
# @Time     : 2020/12/19 17:16
# @Author   : lishijie
import os
import torch
import time
import numpy as np
from pprint import pformat
import torch.nn.functional as F

from utils import setup_logger
from models.main_models import models
import data_loader


class LUPVisQSolver(object):
    """Solver for training and testing LUPVisQNet"""

    def __init__(self, config, path, train_idx, test_idx):
        
        self.epochs = config.epochs
        self.train_patch_num = config.train_patch_num
        self.test_patch_num = config.test_patch_num
        self.dataset = config.dataset
        self.train_index = train_idx
        self.train_test_num = config.train_test_num
        self.batch_size = config.batch_size
        self.sample_num = config.sample_num
        self.class_num = config.class_num
        self.save_model_path = os.path.join(config.save_model_path, config.dataset, 'LUPVisQ')
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        log_name = os.path.join(self.save_model_path, 'train_LUPVisQ.log')
        if os.path.exists(log_name):
            os.remove(log_name)
        self.logger = setup_logger(log_name, 'LUPVisQ')
        self.logger_info(pformat(config))

        self.model_LUPVisQ = models.LUPVisQNet(14, 14, 14, 14, class_num=self.class_num, channel_num=config.channel_num, tau=2).cuda()
        # feature_file = open('./feature.txt', 'w+')
        # for line in list(self.model_LUPVisQ.modules()):
        #     feature_file.writelines(str(line) + '\n')
        self.model_LUPVisQ.train(True)

        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.lr_decay_rate = config.lr_decay_rate
        self.lr_decay_freq = config.lr_decay_freq
        self.weigth_decay = config.weight_decay
        paras = [{'params': self.model_LUPVisQ.parameters(), 'lr': self.lr * self.lrratio},]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weigth_decay)
        # self.solver = torch.optim.SGD(paras, momentum=0.9)
        self.margin_ranking_loss = torch.nn.MarginRankingLoss(margin=0.1)

        train_loader = data_loader.LUPVisQDataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, num_workers=config.num_workers, istrain=True)
        test_loader = data_loader.LUPVisQDataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, config.batch_size, config.num_workers, False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()
        self.logger_info('train with device {} and pytorch {}'.format(0, torch.__version__))
    
    def train(self):
        """Training"""
        best_EMD = 1000.0  # Earth Mover's Distance loss

        for t in range(self.epochs):
            epoch_loss = []
            epoch_emd = []
            epoch_start = time.time()
            batch_start = time.time()
            batch_num = 0

            for img, label in self.train_data:
                batch_num = batch_num + 1
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda(), dtype=torch.float32)
                label = F.softmax(label, dim=-2)

                self.solver.zero_grad()

                res = self.model_LUPVisQ(img, self.sample_num, istrain=True)

                label = label.cuda()
                rank_label = torch.full([res['score_increase1'].size(0), 1], 1).cuda()
                loss = models.LUPVisQ_loss(label, res['score_distribution'], self.margin_ranking_loss, res, rank_label.float())
                emd = models.emd_loss(label, res['score_distribution'])
                if batch_num % 10 == 0:
                    batch_time = time.time() - batch_start
                    batch_start = time.time()
                    self.logger_info(
                        '[{}/{}], batch num: [{}/{}], batch_loss: {:.6f}, Earth Mover\'s Distance: {:.6f}, time: {:.2f}'.format(
                            t+1, self.epochs, batch_num, len(self.train_index) * self.train_patch_num // self.batch_size, loss.item(), emd.item(), batch_time))
                epoch_loss.append(loss.item())
                epoch_emd.append(emd.item())
                loss.backward()
                # grad_file = open('./grad.txt', 'a+')
                # if batch_num % 10 == 0:
                #     for name, parms in self.model_LUPVisQ.named_parameters():	
                #         # print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                #         # '-->grad_value:',parms.grad)
                #         grad_file.writelines('-->name:{}\t-->grad_requirs:{}\t-->grad_value:{}\n'.format(name, parms.requires_grad, parms.grad))
                self.clip_gradient(self.model_LUPVisQ, 1e-1)
                self.solver.step()
            
            test_loss = self.test(self.test_data)
            if test_loss < best_EMD:
                self.logger_info(
                    'Reduce earth mover\'s distance loss from {} to {}'.format(best_EMD, test_loss))
                best_EMD = test_loss
                trained_model_name = 'LUPVisQNet_{}_best_{}.pth'.format(self.dataset, self.train_test_num)
                torch.save(self.model_LUPVisQ.state_dict(), os.path.join(self.save_model_path, trained_model_name))
                self.logger_info(
                    'Save model {} in path {}'.format(trained_model_name, self.save_model_path))
            epoch_time = time.time() - epoch_start
            self.logger_info(
                'Epoch: {}, Train Earth Mover\'s Distance Loss: {}, Test Earth Mover\'s Distance Loss: {}, time: {}'.format(
                    t+1, sum(epoch_emd) / len(epoch_emd), test_loss, epoch_time))
            # update optimizer
            lr = self.lr / pow(10, (t // 6))
            # lr = self.lr * self.lr_decay_rate ** ((t + 1) / self.lr_decay_freq)
            if t > 8:
                self.lrratio = 1
            paras = [{'params': self.model_LUPVisQ.parameters(), 'lr': lr},]
            self.solver = torch.optim.Adam(paras, weight_decay=self.weigth_decay)

            # if (t+1) % 10 == 0:
            #     paras = [{'params': self.model_LUPVisQ.parameters(), 'lr': lr * self.lrratio},]
            #     self.solver = torch.optim.SGD(paras, momentum=0.9)
        
        print('Best test margin ranking loss: %f' % (best_EMD))

        return best_EMD

    def test(self, data):
        """Testing"""
        self.model_LUPVisQ.train(False)
        total_loss = []
        total_emd = []

        with torch.no_grad():
            for img, label in data:
                # Data.
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda(), dtype=torch.float32)
                label = F.softmax(label, dim=-2)

                self.solver.zero_grad()

                res = self.model_LUPVisQ(img, self.sample_num, istrain=False)

                label = label.cuda()
                rank_label = torch.full([res['score_increase1'].size(0), 1], 1).cuda()
                loss = models.LUPVisQ_loss(label, res['score_distribution'], self.margin_ranking_loss, res, rank_label.float())
                emd = models.emd_loss(label, res['score_distribution'])
                total_loss.append(loss.item())
                total_emd.append(emd.item())
        
        self.model_LUPVisQ.train(True)
        return sum(total_emd) / len(total_emd)

    def logger_info(self, s):
        self.logger.info(s)

    def clip_gradient(self, model, clip_value):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)
