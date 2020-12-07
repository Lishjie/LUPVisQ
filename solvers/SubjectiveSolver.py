# -*- coding: utf-8 -*-
# @Time     : 2020/12/04 16:13
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
        self.save_model_path = os.path.join(config.save_model_path, config.dataset, 'subjective')
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.logger = setup_logger(os.path.join(self.save_model_path, 'train_subjective.log'), 'SubjectiveNet')
        self.logger_info(pformat(config))

        self.model_subjective = models.Subjective(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        self.model_subjective.train(True)

        self.margin_ranking_loss = torch.nn.MarginRankingLoss(margin=0.9).cuda()

        backbone_params = list(map(id, self.model_subjective.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_subjective.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weigth_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_subjective.res.parameters(), 'lr': self.lr},
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weigth_decay)

        train_loader = data_loader.SubjectiveDataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, num_workers=config.num_workers, istrain=True)
        test_loader = data_loader.SubjectiveDataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, config.batch_size, config.num_workers, False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()
        self.logger_info('train with device {} and pytorch {}'.format(0, torch.__version__))

    def train(self):
        """Training"""
        best_margin_ranking_loss = 100.0  # margin ranking loss

        for t in range(self.epochs):
            epoch_loss = []
            epoch_start = time.time()
            batch_start = time.time()
            batch_num = 0

            for img_main, img_sub, label in self.train_data:
                batch_num = batch_num + 1
                img_main = torch.tensor(img_main.cuda())
                img_sub = torch.tensor(img_sub.cuda())
                label = torch.tensor(label.cuda())

                self.solver.zero_grad()

                # Generate weights for target network
                paras_main = self.model_subjective(img_main)
                paras_sub = self.model_subjective(img_sub)

                # Building target network
                model_target_main = models.TargetNet(paras_main).cuda()
                for param in model_target_main.parameters():
                    param.requires_grad = False
                model_target_sub = models.TargetNet(paras_sub).cuda()
                for param in model_target_sub.parameters():
                    param.requires_grad = False
                
                # Quality prediction
                score_main = model_target_main(paras_main['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
                score_sub = model_target_sub(paras_sub['target_in_vec'])

                # Margin Ranking Loss
                loss = self.margin_ranking_loss(score_main.squeeze(), score_sub.squeeze(), label.float())
                if batch_num % 100 == 0:
                    batch_time = time.time() - batch_start
                    batch_start = time.time()
                    self.logger_info(
                        '[{}/{}], batch num: [{}/{}], Margin Ranking loss: {:.6f}, time: {:.2f}'.format(
                            t+1, self.epochs, batch_num, len(self.train_index)//2*self.train_patch_num // self.batch_size, loss, batch_time))
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
            
            test_loss = self.test(self.test_data)
            if test_loss < best_margin_ranking_loss:
                self.logger_info(
                    'Reduce margin ranking loss from {} to {}'.format(best_margin_ranking_loss, test_loss))
                best_margin_ranking_loss = test_loss
                trained_model_name = 'SubjectiveNet_{}_best_{}.pth'.format(self.dataset, self.train_test_num)
                torch.save(self.model_subjective.state_dict(), os.path.join(self.save_model_path, trained_model_name))
                self.logger_info(
                    'Save model {} in path {}'.format(trained_model_name, self.save_model_path))
            epoch_time = time.time() - epoch_start
            self.logger_info(
                'Epoch: {}, Train Margin Ranking Loss: {}, Test Margin Ranking Loss: {}, time: {}'.format(
                    t+1, sum(epoch_loss) / len(epoch_loss), test_loss, epoch_time))
            # update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_subjective.res.parameters(), 'lr': self.lr},
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weigth_decay)

        print('Best test margin ranking loss: %f' % (best_margin_ranking_loss))

        return best_margin_ranking_loss
    
    def test(self, data):
        """Testing"""
        self.model_subjective.train(False)
        total_loss = []

        with torch.no_grad():
            for img_main, img_sub, label in data:
                # Data.
                img_main = torch.tensor(img_main.cuda())
                img_sub = torch.tensor(img_sub.cuda())
                label = torch.tensor(label.cuda())

                paras_main = self.model_subjective(img_main)
                paras_sub = self.model_subjective(img_sub)
                model_target_main = models.TargetNet(paras_main).cuda()
                model_target_sub = models.TargetNet(paras_sub).cuda()
                model_target_main.train(False)
                model_target_sub.train(False)
                score_main = model_target_main(paras_main['target_in_vec'])
                score_sub = model_target_sub(paras_sub['target_in_vec'])

                loss = self.margin_ranking_loss(score_main.squeeze(), score_sub.squeeze(), label.float())
                total_loss.append(float(loss.item()))
        
        self.model_subjective.train(True)
        return sum(total_loss) / len(total_loss)

    def logger_info(self, s):
        self.logger.info(s)

