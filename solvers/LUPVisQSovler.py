# -*- coding: utf-8 -*-
# @Time     : 2021/01/03 21:43
# @Author   : lishijie
import os
import time
import torch
import numpy as np
from scipy import stats
from pprint import pformat

import data_loader
from utils import setup_logger
from models.main_models import models


class LUPVisQSolver(object):
    """Solver for training and testing LUPVisQNet"""
    def __init__(self, config, path, train_idx, val_idx):
        # base config
        self.epochs = config.epochs
        self.train_sample_num = config.train_sample_num
        self.val_sample_num = config.val_sample_num
        self.dataset = config.dataset
        self.train_index = train_idx
        self.train_test_num = config.train_test_num
        self.batch_size = config.batch_size
        self.save_model_path = os.path.join(config.save_model_path, config.dataset, 'LUPVisQ')
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        log_name = os.path.join(self.save_model_path, 'train_LUPVisQ.log')
        if os.path.exists(log_name):
            os.remove(log_name)
        self.logger = setup_logger(log_name, 'LUPVisQ')
        self.logger_info(pformat(config))

        # model prepare
        self.repeat_num = config.repeat_num
        self.class_num = config.class_num
        self.channel_num = config.channel_num
        self.tau = config.tau
        self.backbone_type = config.backbone_type
        self.lambda_ = config.lambda_
        self.model_LUPVisQ = models.LUPVisQNet(512, 512, 512, self.class_num, self.backbone_type, self.channel_num, self.tau).cuda()
        # self.model_LUPVisQ.load_state_dict((torch.load('./result/ava_database/LUPVisQ/')))

        # optimizer prepare
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.lr_decay_rate = config.lr_decay_rate
        paras = [{'params': self.model_LUPVisQ.parameters(), 'lr': self.lr},]
        self.optimizer = torch.optim.SGD(paras, momentum=self.momentum, weight_decay=1e-4)

        # dataset prepare
        train_loader = data_loader.LUPVisQDataLoader(config.dataset, path, train_idx, config.patch_size, config.train_sample_num, batch_size=config.batch_size, num_workers=config.num_workers, istrain='train')
        val_loader = data_loader.LUPVisQDataLoader(config.dataset, path, val_idx, config.patch_size, config.val_sample_num, batch_size=config.batch_size, num_workers=config.num_workers, istrain='val')
        self.train_data = train_loader.get_data()
        self.val_data = val_loader.get_data()
        self.logger_info('train with device {} and pytorch {}'.format(0, torch.__version__))

    def train(self):
        """Training"""
        best_EMD = 1000.0  # Earth Mover's Distance loss
        saturates_epoches = 0
        self.model_LUPVisQ.train()

        for t in range(self.epochs):
            epoch_loss = []
            epoch_ACC = []
            epoch_start = time.time()
            batch_start = time.time()
            batch_num = 0

            for img, target in self.train_data:
                batch_num += 1
                img = torch.tensor(img.cuda())
                target = torch.tensor(target.cuda(), dtype=torch.long) - 1
                target = target.flatten()

                self.optimizer.zero_grad()

                # model result
                r_s, g_t, p_t = self.model_LUPVisQ(img, self.repeat_num, istrain='train')

                # to gpu
                target = target.cuda()

                # loss
                loss, prediction = models.LUPVisQ_loss(r_s, target.detach(), g_t, p_t, self.repeat_num, lambda_=self.lambda_)
                num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
                acc = 100.0 * num_corrects / img.size(0)

                # train log
                if batch_num % 100 == 0:
                    batch_time = time.time() - batch_start
                    batch_start = time.time()
                    self.logger_info(
                        '[{}/{}], batch num: [{}/{}], batch_loss: {:.6f}, ACC: {:.3f}, time: {:.2f}'.format(
                            t+1, self.epochs, batch_num, len(self.train_index) * self.train_sample_num // self.batch_size, loss.item(), acc, batch_time))
                epoch_loss.append(loss)
                epoch_ACC.append(loss)

                # backward
                loss.backward()
                self.clip_gradient(self.model_LUPVisQ, 1e-1)
                self.optimizer.step()
            
            # validation
            test_emd, test_lcc_mean, test_srcc_mean, \
                test_lcc_std, test_srcc_std, test_acc = self.val(self.val_data)
            if test_emd < best_EMD:
                self.logger_info(
                    'Reduce earth mover\'s distance loss from {} to {}'.format(best_EMD, test_emd))
                best_EMD = test_emd
                trained_model_name = 'LUPVisQNet_{}_best_{}_{:.6f}.pth'.format(self.dataset, self.train_test_num, best_EMD)
                torch.save(self.model_LUPVisQ.state_dict(), os.path.join(self.save_model_path, trained_model_name))
                self.logger_info(
                    'Save model {} in path {}'.format(trained_model_name, self.save_model_path))
            else:
                saturates_epoches += 1
            epoch_time = time.time() - epoch_start
            self.logger_info(
                'Epoch: {}, Train ACC: {}, Validation Earth Mover\'s Distance Loss: {}, LCC_mean: {:.3f}, SRCC_mean: {:.3f}, LCC_std: {:.3f}, SRCC_std: {:.3f}, ACC: {:.3f}, time: {}'.format(
                    t+1, sum(epoch_ACC) / len(epoch_ACC), test_emd, test_lcc_mean, test_srcc_mean, test_lcc_std, test_srcc_std, test_acc, epoch_time))
            
            # update optimizer
            if saturates_epoches == 5:
                self.lr = self.lr * 0.9
                saturates_epoches = 0
                paras = [{'params': self.model_LUPVisQ.parameters(), 'lr': self.lr},]
                self.optimizer = torch.optim.SGD(paras, momentum=self.momentum, weight_decay=1e-4)
        
        print('Best validation EMD: %f' % (best_EMD))
        return best_EMD
    
    def val(self, data):
        """validation"""
        total_emd = []
        pred_mean = []
        gt_mean = []
        pred_std = []
        gt_std = []
        acc_scores = []

        with torch.no_grad():
            for img, dis, mean, std in data:
                img = torch.tensor(img.cuda())
                dis = torch.tensor(dis.cuda(), dtype=torch.float32)

                self.optimizer.zero_grad()

                # model result
                r_s, g_t, p_t = self.model_LUPVisQ(img, self.repeat_num, istrain='val')

                # calculate prediction distribution
                batch_scores = np.zeros((img.size(0), self.repeat_num), dtype=int)
                batch_dis = np.zeros((img.size(0), self.class_num), dtype=float)
                for index, logits in enumerate(r_s):
                    batch_scores[:, index] = np.array(torch.max(logits, 1)[1].flatten().tolist(), dtype=int)
                for i in range(batch_scores.shape[0]):
                    key = np.unique(batch_scores[i, :])
                    for k in key:
                        mask = (batch_scores[i, :] == k)
                        y = batch_scores[i, :][mask]
                        v = y.size
                        batch_dis[i, int(k)] = float(v)
                batch_dis = batch_dis / np.sum(batch_dis, axis=-1).reshape((-1, 1))  # dim: (batch_size, class_num)

                mean_pred, std_pred = self.cal_mean_std(batch_dis.tolist())
                # mean
                pred_mean += mean_pred
                gt_mean += mean.tolist()
                # std
                pred_std += std_pred
                gt_std += std.tolist()
                # acc
                pred_np = np.array([0 if x <= 5.0 else 1 for x in mean_pred])
                gt_np = np.array([0 if x <= 5.0 else 1 for x in mean.float().cpu().tolist()])
                acc = sum(pred_np == gt_np) / img.size(0)
                acc_scores.append(acc)
                # earth mover's distance
                total_emd.append(self.earth_movers_distance_torch(torch.tensor(batch_dis, dtype=torch.float).unsqueeze(-1), dis).item())
        
        lcc_mean, _ = stats.pearsonr(pred_mean, gt_mean)
        srcc_mean, _ = stats.spearmanr(pred_mean, gt_mean)
        lcc_std, _ = stats.pearsonr(pred_std, gt_std)
        srcc_std, _ = stats.spearmanr(pred_std, gt_std)

        return sum(total_emd) / len(total_emd), lcc_mean, srcc_mean, lcc_std, srcc_std, sum(acc_scores) / len(acc_scores)
    
    def cal_mean_std(self, score_dis):
        mean_batch = []
        std_batch = []

        for item in score_dis:
            mean = 0.0
            std = 0.0

            for score, num in enumerate(item, 1):
                mean += score * num
            for score, num in enumerate(item, 1):
                std += num * (score - mean) ** 2
            std = std ** 0.5
            mean_batch.append(mean)
            std_batch.append(std)
        
        return mean_batch, std_batch

    def earth_movers_distance_torch(self, y, y_pred, r=2):
        cdf_y = torch.cumsum(y, dim=1)
        cdf_pred = torch.cumsum(y_pred, dim=1)
        cdf_diff = cdf_pred - cdf_y
        emd_loss = torch.pow(torch.mean(torch.pow(torch.abs(cdf_diff), r)), 1 / r)
        return emd_loss.mean()

    def clip_gradient(self, model, clip_value):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def logger_info(self, s):
        self.logger.info(s)

