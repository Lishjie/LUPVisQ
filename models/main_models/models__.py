# -*- coding: utf-8 -*-
# @Time     : 2020/12/23 19:52
# @Author   : lishijie
import os
import sys
import pathlib
import torch
from torch import nn
from torch.nn import functional as F
import math
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

from models.backbones import models as models_


class LUPVisQNet(nn.Module):
    """
    Learning to Understand the Perceived Visual Quality of Photos

    Args:
        backbone_out_size: output vector size for BackBoneNet
        sbj_adp_in_size: input vector size for SubjectiveDecisionNet
        sdb_out_size: output vector size for SubjectiveDecisionNet
        channel_num: Multi-dimensional aesthetic channel number
    """
    def __init__(self, backbone_out_size, sbj_adp_in_size, sdb_out_size, class_num, backbone_type='objectiveNet_backbone', channel_num=5, tau=1):
        super(LUPVisQNet, self).__init__()

        self.backbone_out_size = backbone_out_size
        self.sbj_adp_in_size = sbj_adp_in_size
        self.sdb_out_size = sdb_out_size
        self.class_num = class_num
        self.backbone_type = backbone_type
        self.channel_num = channel_num
        self.tau = tau

        # Backbone Network
        self.backboneNet = getattr(models_, self.backbone_type)()

        # Subjective Decision Block
        self.subjectiveDecisionNet = SubjectiveDecisionNet(self.sbj_adp_in_size, self.sbj_adp_in_size*self.channel_num,
                                                           self.sbj_adp_in_size*self.channel_num, self.channel_num, self.channel_num, self.channel_num, self.tau)
        
        # Mulitilayer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(self.sdb_out_size, self.class_num),
            nn.Sigmoid(),
        )

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, img, istrain=True):
        h_i = self.backboneNet(img)

        hidden_h, g_t, score_increase1, score_increase2, \
                score_default, score_decrease1, score_decrease2 = self.subjectiveDecisionNet(h_i, istrain)  # dim: (batch_size, sdb_out_size)

        score_logits_ = self.mlp(hidden_h)
        score_logits = F.softmax(score_logits_, dim=-1)  # dim: (batch_size, class_num)
        # score_logits_onehot = F.one_hot(torch.argmax(score_logits, dim=-1), num_classes=self.class_num).float()

        return score_logits, g_t, score_increase1, score_increase2, score_default, score_decrease1, score_decrease2
    
    def sample_forward(self, img, label_, sample_num=50, istrain=True):
        # initialize tensors
        scores = torch.zeros(sample_num, label_.size(0), self.class_num)

        # sampling
        for time in range(sample_num-1):
            scores[time], _, _, _, _, _, _ = self(img, istrain)
        scores[sample_num-1], g_t, score_increase1, score_increase2, \
            score_default, score_decrease1, score_decrease2 = self(img, istrain)

        # distribution
        score_dis_ = torch.sum(scores, dim=0)
        score_dis = F.softmax(score_dis_, dim=-1)
        score_dis = score_dis.view(score_dis.size(0), score_dis.size(1), 1)
        label = F.softmax(label_, dim=-2)

        # loss
        score_dic = {}
        score_dic['g_t'] = g_t
        score_dic['score_increase1'] = score_increase1
        score_dic['score_increase2'] = score_increase2
        score_dic['score_objective'] = score_default
        score_dic['score_decrease1'] = score_decrease1
        score_dic['score_decrease2'] = score_decrease2
        loss_fn = torch.nn.MarginRankingLoss(margin=0.1)
        rank_label = torch.full([score_increase1.size(0), 1], 1).cuda()

        # to gpu
        score_dis = score_dis.cuda()
        label = label.cuda()

        return LUPVisQ_loss(label.float().detach(), score_dis, loss_fn, score_dic, rank_label, sample_num=sample_num)


class SubjectiveDecisionNet(nn.Module):
    """
    Subjective Decision Block

    Args:
        sd_in_size: input vector size for Subjective Decision Block
        aux_in_size: input vector size for AuxiliaryNet
        att_in_size: input vector size for attention layer
        aux_out_size: output vector size for AuxiliaryNet
        att_out_size: output vector size for attention layer
        tau: temperature for Gumbel-Softmax
    """
    def __init__(self, sd_in_size, aux_in_size, att_in_size, aux_out_size, att_out_size, channel_num=5, tau=1):
        super(SubjectiveDecisionNet, self).__init__()

        self.sd_in_size = sd_in_size
        self.aux_in_size = aux_in_size
        self.att_in_size = att_in_size
        self.aux_out_size = aux_out_size
        self.att_out_size = att_out_size
        self.channel_num = channel_num
        self.tau = tau

        # Multi-dimensional aesthetic channel
        self.linear_increase1 = nn.Sequential(
            nn.Linear(self.sd_in_size, self.sd_in_size),
            nn.ReLU(True),
        )
        self.linear_increase2 = nn.Sequential(
            nn.Linear(self.sd_in_size, self.sd_in_size),
            nn.ReLU(True),
        )
        self.linear_decrease1 = nn.Sequential(
            nn.Linear(self.sd_in_size, self.sd_in_size),
            nn.ReLU(True),
        )
        self.linear_decrease2 = nn.Sequential(
            nn.Linear(self.sd_in_size, self.sd_in_size),
            nn.ReLU(True),
        )

        # Fully Connected Layer & Gate Network
        self.auxiliarynet = AuxiliaryNet(self.aux_in_size, 35, 14, self.channel_num, self.tau)

        # Attention Layer
        self.attention = GatedAttentionNet(self.att_in_size, 35, 14, self.channel_num)

        # Ranking Net
        self.rankingNet = RankingNet(self.sd_in_size, int(self.sd_in_size / 2))

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, h_i, istrain=True):
        # multi-dimensional aesthetic channel
        hidden_increase1 = self.linear_increase1(h_i)  # dim: (batch_size, sd_in_size)
        hidden_increase2 = self.linear_increase2(h_i)
        hidden_descrese1 = self.linear_decrease1(h_i)
        hidden_descrese2 = self.linear_decrease2(h_i)
        linear_h = torch.cat((hidden_increase1, hidden_increase2, h_i, hidden_descrese1, hidden_descrese2), 1)

        # Gate
        g_t = self.auxiliarynet(linear_h, istrain)  # dim: (batch_size, 5)
        alpha = self.attention(linear_h, g_t, istrain)  # dim: (batch_size, 5)

        # attention
        increase1_chn = torch.mul(hidden_increase1, alpha[:, 0].view(hidden_increase1.size(0), 1))
        increase2_chn = torch.mul(hidden_increase2, alpha[:, 1].view(hidden_increase2.size(0), 1))
        default_chn = torch.mul(h_i, alpha[:, 2].view(h_i.size(0), 1))
        decrease1_chn = torch.mul(hidden_descrese1, alpha[:, 3].view(hidden_descrese1.size(0), 1))
        decrease2_chn = torch.mul(hidden_descrese2, alpha[:, 4].view(hidden_descrese2.size(0), 1))
        hidden_h = torch.add(
                      torch.add(
                        torch.add(
                          torch.add(increase1_chn, increase2_chn), default_chn), decrease1_chn), decrease2_chn)
        
        score_increase1 = self.rankingNet(hidden_increase1)
        score_increase2 = self.rankingNet(hidden_increase2)
        score_default = self.rankingNet(h_i)
        score_decrease1 = self.rankingNet(hidden_descrese1)
        score_decrease2 = self.rankingNet(hidden_descrese2)

        return hidden_h, g_t, score_increase1, score_increase2, score_default, score_decrease1, score_decrease2


class AuxiliaryNet(nn.Module):
    """
    Fully Connection Layer & Gate Network

    Args:
        aux_in_size: input vector size for AuxiliaryNet
        aux_out_size: output vector size for AuxiliaryNet
        sample_num: number for sample Gate value
        tau: temperature for Gunbel-Softmax
    """
    def __init__(self, aux_in_size, aux_fc1_size, aux_fc2_size, aux_out_size, tau=1):
        super(AuxiliaryNet, self).__init__()

        self.aux_in_size = aux_in_size
        self.aux_fc1_size = aux_fc1_size
        self.aux_fc2_size = aux_fc2_size
        self.aux_out_size = aux_out_size
        self.tau = tau

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(aux_in_size, aux_fc1_size),
            nn.ReLU(True),
            nn.Linear(aux_fc1_size, aux_fc2_size),
            nn.ReLU(True),
            nn.Linear(aux_fc2_size, aux_out_size),
            nn.ReLU(True),
        )

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, linear_h, istrain=True):
        p_t_ = self.fc(linear_h)
        p_t_ = p_t_.view(p_t_.size(0), p_t_.size(1), 1)  # dim: (batch_size, 5, 1)

        if istrain:
            p_t = p_t_.repeat(1, 1, 2)                   # dim: (batch_size, 5, 2)
            p_t[:, :, 0] = 1 - p_t[:, :, 0]
            g_hat = F.gumbel_softmax(p_t, self.tau, hard=False)
            g_t = g_hat[:, :, 1]
        else:
            m = torch.distributions.bernoulli.Bernoulli(p_t_)
            g_t = m.sample()
        
        return g_t.squeeze()                             # dim: (batch_size, 5)


class GatedAttentionNet(nn.Module):
    """
    Attention Layer after Auxiliary Net
    """
    def __init__(self, att_in_size, att_fc1_size, att_fc2_size, att_out_size):
        super(GatedAttentionNet, self).__init__()

        self.att_in_size = att_in_size
        self.att_fc1_size = att_fc1_size
        self.att_fc2_size = att_fc2_size
        self.att_out_size = att_out_size

        self.fc = nn.Sequential(
            nn.Linear(self.att_in_size, self.att_fc1_size),
            nn.ReLU(True),
            nn.Linear(self.att_fc1_size, self.att_fc2_size),
            nn.ReLU(True),
            nn.Linear(self.att_fc2_size, self.att_out_size),
            nn.ReLU(True),
        )

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    # Use during train
    def gate_softmax(self, logits, gate):
        logits_exp = torch.mul(logits.exp(), gate)
        partition = logits_exp.sum(dim=1, keepdim=True)
        return torch.div(logits_exp, partition)

    # Use during test
    def masked_softmax(self, logits, mask):
        mask_bool = mask <= 0
        logits[mask_bool] = float('-inf')
        return torch.softmax(logits, dim=-1)

    def forward(self, linear_h, g_t, istrain=True):
        e_t = self.fc(linear_h)  # dim: (batch_size, 5)

        if istrain:
            alpha = self.gate_softmax(e_t, g_t)
        else:
            alpha = self.masked_softmax(e_t, g_t)
        
        return alpha.squeeze()  # dim: (batch_size, 5)


class RankingNet(nn.Module):
    """
    Ranking Net for generate every channel's score
    """
    def __init__(self, rk_in_size, rk_fc1_size):
        super(RankingNet, self).__init__()

        self.rk_in_size = rk_in_size
        self.rk_fc1_size = rk_fc1_size

        self.fc = nn.Sequential(
            nn.Linear(self.rk_in_size, self.rk_fc1_size),
            nn.ReLU(True),
            nn.Linear(self.rk_fc1_size, 1),
            nn.ReLU(True),
        )

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, linear_h):
        score = self.fc(linear_h)

        return score


def single_emd_loss(p, q, r=1):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


def emd_loss(p, q, r=1):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch_size × num_classes × 1
        q: estimated distribution of shape mini_batch_size × num_classes × 1
        r: norm parameters
    """
    assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size


def LUPVisQ_loss(p, q, loss_fn, score_dic, rank_label, r=1, scale_loss2=0.01, lambda_=1e-3, sample_num=50):
    emd_loss_ = emd_loss(p, q, r)
    ranking_loss = loss_fn(score_dic['score_increase1'], score_dic['score_increase2'], rank_label) + loss_fn(score_dic['score_increase2'], score_dic['score_objective'], rank_label) \
                   + loss_fn(score_dic['score_objective'], score_dic['score_decrease1'], rank_label) + loss_fn(score_dic['score_decrease1'], score_dic['score_decrease2'], rank_label)
    # T = len(score_dic['g_t'])
    T = score_dic['g_t'].size(1) * sample_num
    regular = (lambda_ * torch.sum(score_dic['g_t'])) / T

    return emd_loss_ + (scale_loss2 * ranking_loss) + regular, emd_loss_
    # return emd_loss_ + (scale_loss2 * ranking_loss)
