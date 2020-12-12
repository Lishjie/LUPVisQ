# -*- coding: utf-8 -*-
# @Time     : 2020/12/12 9:45
# @Author   : lishijie
import os
import torch
import time
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
        self.save_model_path = os.path.join(config.save_model_path, config.dataset, 'LUPVisQ')
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.logger = setup_logger(os.path.join(self.save_model_path, 'train_LUPVisQ.log'), 'LUPVisQ')
        self.logger_info(pformat(config))

        self.model_LUPVisQ = models.LUPVisQNet(80, 80, 80, 10, channel_num=3, tau=1, istrain=True).cuda()
        self.model_LUPVisQ.train(True)

        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weigth_decay = config.weight_decay
        paras = [{'params': self.model_LUPVisQ.parameters(), 'lr': self.lr * self.lrratio},]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weigth_decay)

        train_loader = data_loader.LUPVisQDataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, num_workers=config.num_workers, istrain=True)
        test_loader = data_loader.LUPVisQDataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, config.batch_size, config.num_workers, False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()
        self.logger_info('train with device {} and pytorch {}'.format(0, torch.__version__))
    
    def train(self):
        """Training"""
        best_EMD_loss = 100.0  # Earth Mover's Distance loss

        for t in range(self.epochs):
            epoch_loss = []
            epoch_start = time.time()
            batch_start = time.time()
            batch_num = 0

            for img, label in self.train_data:
                batch_num = batch_num + 1
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())
                label = F.softmax(label, dim=-1)

                self.solver.zero_grad()

                score_dis = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                for _ in range(self.sample_num):
                    output = self.model_LUPVisQ(img)
                    _, score = torch.argmax(output)
                    score_dis[score.tolist()[0]] += 1
                score_dis_tensor = torch.tensor(score_dis)
                score_dis_tensor = F.softmax(score_dis_tensor, dim=-1)

                loss = models.single_emd_loss(label.float(), score_dis_tensor, r=1)
                if batch_num % 100 == 0:
                    batch_time = time.time() - batch_start
                    batch_start = time.time()
                    self.logger_info(
                        '[{}/{}], batch num: [{}/{}], Earth Mover\'s Distance loss: {:.6f}, time: {:.2f}'.format(
                            t+1, self.epochs, batch_num, len(self.train_index)//2*self.train_patch_num // self.batch_size, loss, batch_time))
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
            
            test_loss = self.test(self.test_data)
            if test_loss < best_EMD_loss:
                self.logger_info(
                    'Reduce earth mover\'s distance loss from {} to {}'.format(best_EMD_loss, test_loss))
                best_EMD_loss = test_loss
                trained_model_name = 'LUPVisQNet_{}_best_{}.pth'.format(self.dataset, self.train_test_num)
                torch.save(self.model_LUPVisQ.state_dict(), os.path.join(self.save_model_path, trained_model_name))
                self.logger_info(
                    'Save model {} in path {}'.format(trained_model_name, self.save_model_path))
            epoch_time = time.time() - epoch_start
            self.logger_info(
                'Epoch: {}, Train Earth Mover\'s Distance Loss: {}, Test Margin Ranking Loss: {}, time: {}'.format(
                    t+1, sum(epoch_loss) / len(epoch_loss), test_loss, epoch_time))
            # update optimizer
            lr = self.lr / pow(10, (t // 6))
            if t > 8:
                self.lrratio = 1
            paras = [{'params': self.model_LUPVisQ.parameters(), 'lr': lr * self.lrratio},]
            self.solver = torch.optim.Adam(paras, weight_decay=self.weigth_decay)
        
        print('Best test margin ranking loss: %f' % (best_EMD_loss))

        return best_EMD_loss

    def test(self, data):
        """Testing"""
        self.model_LUPVisQ.train(False)
        total_loss = []

        with torch.no_grad():
            for img, label in data:
                # Data.
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())
                label = F.softmax(label, dim=-1)

                score_dis = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                for _ in range(self.sample_num):
                    output = self.model_LUPVisQ(img)
                    _, score = torch.argmax(output)
                    score_dis[score.tolist()[0]] += 1
                score_dis_tensor = torch.tensor(score_dis)
                score_dis_tensor = F.softmax(score_dis_tensor, dim=-1)

                loss = models.single_emd_loss(label.float(), score_dis_tensor, r=1)
                total_loss.append(float(loss.item()))
        
        self.model_LUPVisQ.train(True)
        return sum(total_loss) / len(total_loss)

    def logger_info(self, s):
        self.logger.info(s)

