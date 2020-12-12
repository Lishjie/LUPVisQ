# -*- coding: utf-8 -*-
# @Time     : 2020/12/10 15:48
# @Author   : lishijie
import os
import sys
import pathlib
import torch
from torch import nn
import torch.nn.functional as F
import math
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

from models.backbones import models as models_


class LUPVisQNet(nn.Module):
    """
    Learning to Understand the Perceived Visual Quality of Photos

    Args:
        obj_out_size: output vector size for ObjectiveNet
        sbj_out_size: output vector size for SubjectiveNet
        sdb_out_size: output vector size for SubjectiveDecisionNet
        fc_in_size: input vector size for Fully Connected Block
        channel_num: Multi-dimensional aesthetic channel number

    Example:
        LUPVisQNet(80, 80, 80, 10)
    """
    def __init__(self, obj_out_size, sbj_out_size, sdb_out_size, class_num, channel_num=3, tau=1, istrain=True):
        super(LUPVisQNet, self).__init__()

        self.obj_out_size = obj_out_size
        self.sbj_out_size = sbj_out_size
        self.sdb_out_size = sdb_out_size
        self.fc_in_size = obj_out_size + sdb_out_size
        self.class_num = class_num
        self.tau = tau
        self.istrain = istrain

        # ObjectiveNet
        self.objectiveNet = models_.objectiveNet_backbone()

        # SubjectiveNet
        self.subjectiveNet = models_.subjectiveNet_backbone()

        # Subjective Decision Block
        self.subjectiveDecisionNet = SubjectiveDecisionNet(sbj_out_size, (obj_out_size+sbj_out_size)*3, (obj_out_size+sbj_out_size)*3,
                                                           channel_num, channel_num, sbj_out_size, tau, istrain)
        
        # Mulitilayer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(self.fc_in_size, int(self.fc_in_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.fc_in_size / 2), int(self.fc_in_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.fc_in_size / 4), class_num),
            nn.Sigmoid()
        )

        # initialize
        for m_name in self._modules:
            if isinstance(m_name, nn.Linear):
                nn.init.normal_(m_name.weight, 0, 0.01)
                nn.init.normal_(m_name.bias, 0)

    def forward(self, img):
        # objectiveNet backbone
        hi_o = self.objectiveNet(img)
        # subjective backbone
        params_sub = self.subjectiveNet(img)
        targetNet = models_.TargetNet(params_sub)
        for param in targetNet.parameters():
            param.requires_grad = False
        hi_s = targetNet(params_sub['target_in_vec'])
        # Subjective Decision Block
        hi_s = self.subjectiveDecisionNet(hi_o, hi_s)
        # Mulitilayer perceptron
        hi = hi_o + hi_s
        hi = nn.Sigmoid(hi)
        logits = self.mlp(hi)
        logits_softmax = F.softmax(logits, dim=-1)
        return logits_softmax


class SubjectiveDecisionNet(nn.Module):
    """
    Subjective Decision Block

    Args:
        sd_in_size: input vector size for Subjective Decision Block
        aux_in_size: input vector size for AuxiliaryNet
        att_in_size: input vector size for attention layer
        att_out_size: output vector size for attention layer
        aux_out_size: output vector size for AuxiliaryNet
        sd_out_size: output vector size for Subjective Decision Block
        tau: temperature for Gumbel-Softmax

    Example:
        SubjectiveDecisionNet(80, 480, 480, 3, 3, 80, True)
    """
    def __init__(self, sd_in_size, aux_in_size, att_in_size, aux_out_size, att_out_size, sd_out_size, tau=1, istrain=True):
        super(SubjectiveDecisionNet, self).__init__()
        
        self.sd_in_size = sd_in_size
        self.aux_in_size = aux_in_size
        self.att_in_size = att_in_size
        self.aux_out_size = aux_out_size
        self.att_out_size = att_out_size
        self.sd_out_size = sd_out_size

        # Multi-dimensional aesthetic channel
        self.linear1 = nn.Sequential(
            nn.Linear(sd_in_size, sd_in_size),
            nn.Sigmoid(),
            # self-attention
            # dot product g1
        )
        self.linear2 = nn.Sequential(
            nn.Linear(sd_in_size, sd_in_size),
            nn.Sigmoid(),
            # self-attention
            # dot product g2
        )
        self.linear3 = nn.Sequential(
            nn.Linear(sd_in_size, sd_in_size),
            nn.Sigmoid(),
            # self-attention
            # dot product g3
        )

        # Fully Connected Layer & Gate Network
        self.auxiliarynet = AuxiliaryNet(sd_in_size, aux_in_size, aux_out_size, tau, istrain)

        # Attention Layer
        self.attention = nn.Sequential(
            nn.Linear(att_in_size, int(att_in_size / 2)),  # 480, 240
            nn.ReLU(),
            nn.Linear(int(att_in_size / 2), int(att_in_size / 4)),  # 240, 120
            nn.ReLU(),
            nn.Linear(int(att_in_size / 4), int(att_in_size / 8)),  # 120, 60
            nn.ReLU(),
            nn.Linear(int(att_in_size / 8), att_out_size),  # 60, 3
            nn.Sigmoid(),
        )

        # initialize
        for m_name in self._modules:
            if isinstance(m_name, nn.Linear):
                nn.init.normal_(m_name.weight, 0, 0.01)
                nn.init.normal_(m_name.bias, 0)

    
    def selfAttention(query, key, value):
        """Compute 'Scaled Dot Product Attention'"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_atten = F.softmax(scores, dim=-1)
        return torch.matmul(p_atten, value), p_atten  # hi_a, attention_value
    
    def masked_softmax(self, logits, mask):
        mask_bool = mask > 0
        logits[mask_bool] = float('-inf')
        return torch.softmax(logits, dim=1)

    def forward(self, h_io, h_is):
        # multi-dimensional aesthetic channel
        linear_h1 = self.linear1(h_is)  # channel1 dim: (batch_size, 80)
        h1_attention = self.selfAttention(linear_h1, linear_h1, linear_h1)
        linear_h1 = torch.cat((h_io, linear_h1), 1)  # dim: (batch_size, 160)
        linear_h2 = self.linear2(h_is)  # channel2
        h2_attention = self.selfAttention(linear_h2, linear_h2, linear_h2)
        linear_h2 = torch.cat((h_io, linear_h2), 1)
        linear_h3 = self.linear3(h_is)  # channel3
        h3_attention = self.selfAttention(linear_h3, linear_h3, linear_h3)
        linear_h3 = torch.cat((h_io, linear_h3), 1)
        linear_h = torch.cat((linear_h1, linear_h2, linear_h3), 1)  # dim: (batch_size, 480)

        # fully connected layer & gate network
        gt = self.auxiliarynet(linear_h)  # dim: (batch_size, 3)

        # attention
        e_t = self.attention(linear_h)
        alpha = self.masked_softmax(e_t, gt)  # dim: (batch_size, 3)
        h1_attention = h1_attention * alpha[:, 0]
        h2_attention = h2_attention * alpha[:, 1]
        h3_attention = h3_attention * alpha[:, 2]

        # add
        hi_s = h1_attention + h2_attention + h3_attention
        hi_s = nn.Sigmoid(hi_s)
        return hi_s  # dim: (batch_size, 80)


class AuxiliaryNet(nn.Module):
    """
    Fully Connected Layer & Gate Network

    Args:
        aux_in_size: input vector size for AuxiliaryNet
        aux_out_size: output vector size for AuxiliaryNet
        tau: temperature for Gumbel-Softmax
    """
    def __init__(self, sd_in_size, aux_in_size, aux_out_size, tau=1, istrain=True):
        super(AuxiliaryNet, self).__init__()

        self.sd_in_size = sd_in_size
        self.aux_in_size = aux_in_size
        self.aux_out_size = aux_out_size
        self.tau = tau
        self.istrain = istrain

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(aux_in_size, int(aux_in_size / 2)),  # 480, 240
            nn.ReLU(),
            nn.Linear(int(aux_in_size / 2), int(aux_in_size / 4)),  # 240, 120
            nn.ReLU(),
            nn.Linear(int(aux_in_size / 4), int(aux_in_size / 8)),  # 120, 60
            nn.ReLU(),
            nn.Linear(int(aux_in_size / 8), aux_out_size),  # 60, 3
            nn.Sigmoid(),
        )

        # initialize
        for m_name in self._modules:
            if isinstance(m_name, nn.Linear):
                nn.init.normal_(m_name.weight, 0, 0.01)
                nn.init.normal_(m_name.bias, 0)

    def forward(self, linear_h):
        pt  = self.fc(linear_h)
        pt = pt.view(pt.size(0), pt.size(1), 1)  # dim: (batch_size, 3, 1)
        
        if self.istrain:
            pt = pt.repeat(1, 1, 2)              # dim: (batch_size, 3, 2)
            pt[:, :, 0] = 1 - pt[:, :, 0]
            g_hat = F.gumbel_softmax(pt, self.tau, hard=False)
            gt = g_hat[:, :, 1]
        else:
            m = torch.distributions.bernoulli.Bernoulli(pt)
            gt = m.sample()
        
        return gt.squeeze()  # dim: (batch_size, 3)

def single_emd_loss(p, q, r=1):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes x 1
        q: estimate distance distribution of shape num_classes x 1
        r: norm parameter 
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i], q[:i])) ** r
    return (emd_loss / length) ** (1. / r)

def emd_loss(p, q, r=1):
    """
    Earth Mover's Distance on a batch

    Args:
        p: true distribution of shape mini_batch x num_classes x 1
        q: estimated distribution of shape mini_batch x num_classes x 1
        r: norm parameters
    """
    assert p.shape == q.shape, "shape of the distribution batches must be the same."
    mini_batch_size = p.shape[0]
    loss_vector = []
    for i in range(mini_batch_size):
        loss_vector.append(single_emd_loss(p[i], q[i], r=r))
    return sum(loss_vector) / mini_batch_size
