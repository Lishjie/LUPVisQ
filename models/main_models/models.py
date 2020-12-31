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
import torchvision.models as torch_models

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
    
    def forward(self, img, sample_num=1, istrain=True):
        h_i = self.backboneNet(img)

        out = self.subjectiveDecisionNet(h_i, sample_num, istrain)  # dim: (batch_size, sdb_out_size) or (batch, sample_num, sdb_out_size)

        score_list = []
        # score_total = torch.zeros(sample_num, img.size(0), self.class_num)
        if sample_num > 1:
            for time in range(sample_num):
                score_logits_ = self.mlp(out['hidden_h'][:, time, :].squeeze())
                score_logits = F.softmax(score_logits_, dim=-1)  # dim: (batch_size, class_num)
                # score_logits_onehot = F.one_hot(torch.argmax(score_logits, dim=-1), num_classes=self.class_num)
                score_list.append(torch.flatten(score_logits))  # [dim: (batch_size, 10)] * 50
                # score_total[time] = score_logits_onehot
        else:
            score_logits_ = self.mlp(out['hidden_h'].squeeze())
            score_logits = F.softmax(score_logits_, dim=-1)  # dim: (batch_size, class_num)
            # score_logits_onehot = F.one_hot(torch.argmax(score_logits, dim=-1), num_class=self.class_num)
            score_list.append(torch.flatten(score_logits))
            # score_total[0] = score_logits_onehot
        score_distribution_ = torch.cat(tuple([x for x in score_list]), 0)
        score_distribution_ = score_distribution_.view(sample_num, -1)  # dim: (sample_num, -1)
        score_distribution = torch.sum(score_distribution_, 0)
        score_distribution = score_distribution.view(h_i.size(0), self.class_num, 1)
        score_distribution_softmax = F.softmax(score_distribution.float(), dim=-2)

        res = {}
        res['score_distribution'] = score_distribution_softmax
        # res['score_total'] = torch.sum(score_total, dim=0)  # dim: (batch_size, class_num)
        res['g_t'] = out['g_t']
        res['score_increase1'] = out['score_increase1']
        res['score_increase2'] = out['score_increase2']
        res['score_objective'] = out['score_objective']
        res['score_decrease1'] = out['score_decrease1']
        res['score_decrease2'] = out['score_decrease2']

        return res


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
    
    def forward(self, h_i, sample_num=1, istrain=True):
        # multi-dimensional aesthetic channel
        hidden_increase1 = self.linear_increase1(h_i)  # dim: (batch_size, sd_in_size)
        hidden_increase2 = self.linear_increase2(h_i)
        hidden_descrese1 = self.linear_decrease1(h_i)
        hidden_descrese2 = self.linear_decrease2(h_i)
        linear_h = torch.cat((hidden_increase1, hidden_increase2, h_i, hidden_descrese1, hidden_descrese2), 1)

        # gate
        g_t = self.auxiliarynet(linear_h, sample_num, istrain)  # dim: (batch_size, 5, sample_num)
        alpha = self.attention(linear_h, g_t, sample_num, istrain)  # dim: (batch_size, 5, sample_num)

        # attention
        alpha_chn1 = alpha[:, 0, :].squeeze()
        alpha_chn2 = alpha[:, 1, :].squeeze()
        alpha_chn3 = alpha[:, 2, :].squeeze()
        alpha_chn4 = alpha[:, 3, :].squeeze()
        alpha_chn5 = alpha[:, 4, :].squeeze()
        if sample_num > 1:
            increase1_chn_ = hidden_increase1.view(hidden_increase1.size(0), 1, hidden_increase1.size(1))
            increase1_chn = increase1_chn_.repeat(1, sample_num, 1)  # dim: (batch_size, sample_num, sd_in_size)
            alpha_chn1 = alpha_chn1.view(alpha_chn1.size(0), alpha_chn1.size(1), 1)
            increase2_chn_ = hidden_increase2.view(hidden_increase2.size(0), 1, hidden_increase2.size(1))
            increase2_chn = increase2_chn_.repeat(1, sample_num, 1)
            alpha_chn2 = alpha_chn2.view(alpha_chn2.size(0), alpha_chn2.size(1), 1)
            objective_chn_ = h_i.view(h_i.size(0), 1, h_i.size(1))
            objective_chn = objective_chn_.repeat(1, sample_num, 1)
            alpha_chn3 = alpha_chn3.view(alpha_chn3.size(0), alpha_chn3.size(1), 1)
            descrese1_chn_ = hidden_descrese1.view(hidden_descrese1.size(0), 1, hidden_descrese1.size(1))
            descrese1_chn = descrese1_chn_.repeat(1, sample_num, 1)
            alpha_chn4 = alpha_chn4.view(alpha_chn4.size(0), alpha_chn4.size(1), 1)
            descrese2_chn_ = hidden_descrese2.view(hidden_descrese2.size(0), 1, hidden_descrese2.size(1))
            descrese2_chn = descrese2_chn_.repeat(1, sample_num, 1)
            alpha_chn5 = alpha_chn5.view(alpha_chn5.size(0), alpha_chn5.size(1), 1)
        else:
            increase1_chn = hidden_increase1
            increase2_chn = hidden_increase2
            objective_chn = h_i
            descrese1_chn = hidden_descrese1
            descrese2_chn = hidden_descrese2
        increase1_chn_att = torch.mul(increase1_chn, alpha_chn1)
        increase2_chn_att = torch.mul(increase2_chn, alpha_chn2)
        objective_chn_att = torch.mul(objective_chn, alpha_chn3)
        descrese1_chn_att = torch.mul(descrese1_chn, alpha_chn4)
        descrese2_chn_att = torch.mul(descrese2_chn, alpha_chn5)
        hidden_h = increase1_chn_att + increase2_chn_att \
            + objective_chn_att + descrese1_chn_att + descrese2_chn_att
        
        score_increase1 = self.rankingNet(hidden_increase1)
        score_increase2 = self.rankingNet(hidden_increase2)
        score_objective = self.rankingNet(h_i)
        score_decrease1 = self.rankingNet(hidden_descrese1)
        score_decrease2 = self.rankingNet(hidden_descrese2)

        out = {}
        out['hidden_h'] = hidden_h
        out['g_t'] = g_t
        out['score_increase1'] = score_increase1
        out['score_increase2'] = score_increase2
        out['score_objective'] = score_objective
        out['score_decrease1'] = score_decrease1
        out['score_decrease2'] = score_decrease2

        return out


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

    def forward(self, linear_h, sample_num=1, istrain=True):
        p_t_ = self.fc(linear_h)
        p_t_ = p_t_.view(p_t_.size(0), p_t_.size(1), 1)  # dim: (batch_size, 5, 1)

        if istrain:
            p_t = p_t_.repeat(1, 1, 2)                # dim: (batch_size, 5, 2)
            p_t[:, :, 0] = 1 - p_t[:, :, 0]
            g_hat = F.gumbel_softmax(p_t, self.tau, hard=False)
            g_t_ = g_hat[:, :, 1]
            if sample_num > 1:
                g_t_ = g_t_.unsqueeze(-1)
                g_t = g_t_.repeat(1, 1, sample_num)   # dim: (batch_size, 5, sample_num)
                for time in range(1, sample_num):
                    g_hat = F.gumbel_softmax(p_t, self.tau, hard=False)
                    g_t[:, :, time] = g_hat[:, :, 1]
            else:
                g_t = g_t_
        else:
            m = torch.distributions.bernoulli.Bernoulli(p_t_)
            g_t_ = m.sample()
            if sample_num > 1:
                g_t = g_t_.repeat(1, 1, sample_num)
                for time in range(1, sample_num):
                    temp_t = m.sample()
                    g_t[:, :, time] = temp_t[:, :, 0]
            else:
                g_t = g_t_
        
        return g_t  # dim: (batch_size, 5, sample_num)


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
        # gate_ = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        # gate_ = gate_.unsqueeze(-1).repeat(1, 2)
        # gate_ = gate_.unsqueeze(0).repeat(2, 1, 1)
        logits_exp = torch.mul(logits.exp(), gate)
        partition = logits_exp.sum(dim=1, keepdim=True)
        return torch.div(logits_exp, partition)
        # return gate_.cuda()

    # Use during test
    def masked_softmax(self, logits, mask):
        mask_bool = mask > 0
        logits[mask_bool] = float('-inf')
        return torch.softmax(logits, dim=-1)

    def forward(self, linear_h, g_t, sample_num=1, istrain=True):
        e_t_ = self.fc(linear_h)  # dim: (batch_size, 5)

        if istrain:
            e_t_ = e_t_.view(e_t_.size(0), e_t_.size(1), 1)
            if sample_num > 1:
                e_t = e_t_.repeat(1, 1, sample_num)
            else:
                e_t = e_t_
            alpha = self.gate_softmax(e_t, g_t)
        else:
            e_t_ = e_t_.view(e_t_.size(0), e_t_.size(1), 1)
            if sample_num > 1:
                e_t = e_t_.repeat(1, 1, sample_num)
            else:
                e_t = e_t_
            alpha = self.masked_softmax(e_t, g_t)
        
        return alpha


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

def earth_movers_distance_torch(y, y_pred, r=2):
    cdf_y = torch.cumsum(y, dim=1)
    cdf_pred = torch.cumsum(y_pred, dim=1)
    cdf_diff = cdf_pred - cdf_y
    emd_loss = torch.pow(torch.mean(torch.pow(torch.abs(cdf_diff), r)), 1 / r)
    return emd_loss.mean()

def LUPVisQ_loss(p, q, loss_fn, score_dic, rank_label, r=2, sample_num=50, scale_loss1=1, scale_loss2=1e-3, lambda_=1e-3):
    # emd_loss_ = emd_loss(p, q, r)
    emd_loss_ = earth_movers_distance_torch(p, q, r)
    ranking_loss = loss_fn(score_dic['score_increase1'], score_dic['score_increase2'], rank_label) + loss_fn(score_dic['score_increase2'], score_dic['score_objective'], rank_label) \
                   + loss_fn(score_dic['score_objective'], score_dic['score_decrease1'], rank_label) + loss_fn(score_dic['score_decrease1'], score_dic['score_decrease2'], rank_label)
    # T = len(score_dic['g_t'])
    T = score_dic['g_t'].size(1) * sample_num
    regular = (lambda_ * torch.sum(score_dic['g_t'])) / T

    return (scale_loss1 * emd_loss_) + (scale_loss2 * ranking_loss) + regular
    # return emd_loss_ + (scale_loss2 * ranking_loss)
