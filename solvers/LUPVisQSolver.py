# -*- coding: utf-8 -*-
# @Time     : 2020/12/26 10:10
# @Author   : lishijie
import os
import torch
import time
import numpy as np
from pprint import pformat
from scipy import stats
import torch.nn.functional as F

from utils import setup_logger
from models.main_models import models
import data_loader

class LUPVisQSolver(object):
    """Solver for training and testing LUPVisQNet"""

    def __init__(self, config, path, train_idx, test_idx):
        
        # base config
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

        # model prepare
        self.model_LUPVisQ = models.LUPVisQNet(14, 14, 14, class_num=self.class_num, backbone_type=config.backbone, channel_num=config.channel_num, tau=0.5).cuda()
        self.model_LUPVisQ.train(True)

        # optim prepare
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.lr_decay_rate = config.lr_decay_rate
        self.lr_decay_freq = config.lr_decay_freq
        self.weigth_decay = config.weight_decay
        paras = [{'params': self.model_LUPVisQ.parameters(), 'lr': self.lr * self.lrratio},]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weigth_decay)
        self.margin_ranking_loss = torch.nn.MarginRankingLoss(margin=0.7)

        # dataset prepare
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
            pred_mean = []
            gt_mean = []
            pred_std = []
            gt_std = []
            epoch_start = time.time()
            batch_start = time.time()
            batch_num = 0

            for img, label, mean, std in self.train_data:
                batch_num = batch_num + 1
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda(), dtype=torch.float32)

                # label = F.softmax(label, dim=-2)

                self.solver.zero_grad()

                res = self.model_LUPVisQ(img, self.sample_num, istrain=True)

                # to gpu
                label = label.cuda()
                rank_label = torch.full([res['score_increase1'].size(0), 1], 1).cuda()
                
                # loss
                loss = models.LUPVisQ_loss(label.float().detach(), res['score_distribution'], self.margin_ranking_loss, res, rank_label.float(), r=2)
                emd = models.earth_movers_distance_torch(label.float().detach(), res['score_distribution'], r=2)
                # emd = models.emd_loss(label, res['score_distribution'])
                mean_pred, std_pred = self.cal_mean_std(res['score_distribution'].squeeze().cpu().tolist())
                epoch_loss.append(loss.item())
                epoch_emd.append(emd.item())

                # mean
                pred_mean = pred_mean + mean_pred
                gt_mean = gt_mean + mean.float().cpu().tolist()

                # std
                pred_std = pred_std + std_pred
                gt_std = gt_std + std.float().cpu().tolist()

                # acc
                pred_np = np.array([0 if x <= 5.0 else 1 for x in mean_pred])
                gt_np = np.array([0 if x <= 5.0 else 1 for x in mean.float().cpu().tolist()])
                acc = sum(pred_np == gt_np) / img.size(0)

                # train log
                if batch_num % 100 == 0:
                    batch_time = time.time() - batch_start
                    batch_start = time.time()
                    lcc_mean, _ = stats.pearsonr(pred_mean, gt_mean)
                    srcc_mean, _ = stats.spearmanr(pred_mean, gt_mean)
                    lcc_std, _ = stats.pearsonr(pred_std, gt_std)
                    srcc_std, _ = stats.spearmanr(pred_std, gt_std)
                    self.logger_info(
                        '[{}/{}], batch num: [{}/{}], batch_loss: {:.6f}, Earth Mover\'s Distance: {:.6f}, LCC_mean: {:.3f}, SRCC_mean: {:.3f}, LCC_std: {:.3f}, SRCC_std: {:.3f}, ACC: {:.3f}, time: {:.2f}'.format(
                            t+1, self.epochs, batch_num, len(self.train_index) * self.train_patch_num // self.batch_size, loss.item(), emd.item(), lcc_mean, srcc_mean, lcc_std, srcc_std, acc, batch_time))
                loss.backward()
                self.clip_gradient(self.model_LUPVisQ, 1e-1)
                self.solver.step()
        
            # test
            test_emd, test_lcc_mean, test_srcc_mean, \
                test_lcc_std, test_srcc_std, test_acc = self.test(self.test_data)
            if test_emd < best_EMD:
                self.logger_info(
                    'Reduce earth mover\'s distance loss from {} to {}'.format(best_EMD, test_emd))
                best_EMD = test_emd
                trained_model_name = 'LUPVisQNet_{}_best_{}_{:.3f}.pth'.format(self.dataset, self.train_test_num, best_EMD)
                torch.save(self.model_LUPVisQ.state_dict(), os.path.join(self.save_model_path, trained_model_name))
                self.logger_info(
                    'Save model {} in path {}'.format(trained_model_name, self.save_model_path))
            epoch_time = time.time() - epoch_start
            self.logger_info(
                'Epoch: {}, Train Earth Mover\'s Distance Loss: {}, Test Earth Mover\'s Distance Loss: {}, LCC_mean: {:.3f}, SRCC_mean: {:.3f}, LCC_std: {:.3f}, SRCC_std: {:.3f}, ACC: {:.3f}, time: {}'.format(
                    t+1, sum(epoch_emd) / len(epoch_emd), test_emd, test_lcc_mean, test_srcc_mean, test_lcc_std, test_srcc_std, test_acc, epoch_time))

            # update optimizer
            lr = self.lr / pow(10, (t // 6))
            # lr = self.lr * self.lr_decay_rate ** ((t + 1) / self.lr_decay_freq)
            if t > 8:
                self.lrratio = 1
            paras = [{'params': self.model_LUPVisQ.parameters(), 'lr': lr},]
            self.solver = torch.optim.Adam(paras, weight_decay=self.weigth_decay)
        
        print('Best test margin ranking loss: %f' % (best_EMD))
    
    def test(self, data):
        """Testing"""
        self.model_LUPVisQ.train(False)
        total_loss = []
        total_emd = []
        pred_mean = []
        gt_mean = []
        pred_std = []
        gt_std = []
        acc_scores = []

        with torch.no_grad():
            for img, label, mean, std in data:
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda(), dtype=torch.float32)
                # label = F.softmax(label, dim=-2)

                res = self.model_LUPVisQ(img, self.sample_num, istrain=True)

                # to gpu
                label = label.cuda()
                rank_label = torch.full([res['score_increase1'].size(0), 1], 1).cuda()

                # loss
                # loss = models.LUPVisQ_loss(label.float().detach(), res['score_distribution'], self.margin_ranking_loss, res, rank_label.float())
                loss = models.earth_movers_distance_torch(label.float().detach(), res['score_distribution'], r=1)
                # emd = models.emd_loss(label, res['score_distribution'])
                mean_pred, std_pred = self.cal_mean_std(res['score_distribution'].squeeze().cpu().tolist())
                total_loss.append(loss.item())
                total_emd.append(loss.item())

                # mean
                pred_mean = pred_mean + mean_pred
                gt_mean = gt_mean + mean.float().cpu().tolist()

                # std
                pred_std = pred_std + std_pred
                gt_std = gt_std + std.float().cpu().tolist()

                # acc
                pred_np = np.array([0 if x <= 5.0 else 1 for x in mean_pred])
                gt_np = np.array([0 if x <= 5.0 else 1 for x in mean.float().cpu().tolist()])
                acc = sum(pred_np == gt_np) / img.size(0)
                acc_scores.append(acc)
        
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

    def logger_info(self, s):
        self.logger.info(s)

    def clip_gradient(self, model, clip_value):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

