# -*- coding: utf-8 -*-
# @Time     : 2021/01/02 16:55
# @Author   : lishijie
import os
import sys
import pathlib
import torch
import numpy as np
from torch import cat, nn
from torch.nn import functional as F
import math
from torch.nn.modules import linear
import torchvision.models as torch_models

__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))


class LUPVisQNet(nn.Module):
    """
    Learning to Understand the Perceived Visual Quality of Photos

    Args:
        backbone_out_size: output vector size from backbone
        sbj_in_size: input vector for SubjectiveDecisionNet
        sbj_out_size: output vector from SubjectiveDecisionNet
        backbone_type: backbone type
    """
    def __init__(self, backbone_out_size, sbj_in_size, sbj_out_size, class_num, backbone_type='inceptionv3_torchmodel', channel_num=5, tau=0.1):
        super(LUPVisQNet, self).__init__()

        self.backbone_out_size = backbone_out_size
        self.sbj_in_size = sbj_in_size
        self.sbj_out_size = sbj_out_size
        self.class_num = class_num
        self.backbone_type = backbone_type
        self.channel_num = channel_num
        self.tau = tau
        
        # Backbone Network
        self.backboneNet = getattr(self, self.backbone_type)(pretrained=True, num_classes=1000)
        self.backbone_output_adapter = nn.Sequential(
            nn.Linear(1000, self.backbone_out_size),
            nn.ReLU(True),
        )

        # Subjective Decision Block
        self.subjectiveDecisionNet = SubjectiveDecisionNet(self.sbj_in_size, self.sbj_in_size, sbj_out_size, 
                                                            aux_out_size=1, channel_num=5, tau=0.1)

        # Mulitilayer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(self.sbj_out_size, self.class_num),
            nn.ReLU(True),
            nn.Softmax(),
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
    
    def inceptionv3_torchmodel(self, pretrained=True, num_classes=1000):
        return torch_models.inception_v3(pretrained=pretrained, num_classes=num_classes)

    def resnet50_torchmodel(self, pretrained=True, num_classes=512):
        return torch_models.resnet50(pretrained=pretrained, num_classes=num_classes)

    def forward(self, img, repeat_num=1000, istrain='train'):
        r_b_ = self.backboneNet(img)
        r_b = self.backbone_output_adapter(r_b_[0])
        r_g, g_t, p_t = self.subjectiveDecisionNet(r_b, repeat_num, istrain)
        r_s = []

        for time in range(repeat_num):
            r_s.append(self.mlp(r_g[:, :, time].squeeze()))

        return r_s, g_t, p_t


class SubjectiveDecisionNet(nn.Module):
    """
    Subjective Decision Block

    Args:
        sbj_in_size: input vector size for Subjective Decision Block
        aux_in_size: input vector size for AuxiliaryNet
        sbj_out_size: output vector size for Subjective Decision Block
        aux_out_size: output vector size for AuxiliaryNet
        tau: temperature for Gumbel-Softmax
    """
    def __init__(self, sbj_in_size, aux_in_size, sbj_out_size, aux_out_size, channel_num=5, tau=0.1):
        super(SubjectiveDecisionNet, self).__init__()

        self.sbj_in_size = sbj_in_size
        self.aux_in_size = aux_in_size
        self.sbj_out_size = sbj_out_size
        self.aux_out_size = aux_out_size
        self.channel_num = channel_num
        self.tau = tau

        # Multi-dimensional aesthetic channel
        # self.linear_channel = nn.Sequential(
        #     nn.Linear(self.sbj_in_size, self.sbj_in_size),
        #     nn.ReLU(True),
        # )
        self.linear_channel_increase1 = nn.Sequential(
            nn.Linear(self.sbj_in_size, self.sbj_in_size),
            nn.ReLU(True),
        )
        self.linear_channel_increase2 = nn.Sequential(
            nn.Linear(self.sbj_in_size, self.sbj_in_size),
            nn.ReLU(True),
        )
        self.linear_channel_default = nn.Sequential(
            nn.Linear(self.sbj_in_size, self.sbj_in_size),
            nn.ReLU(True),
        )
        self.linear_channel_decrease1 = nn.Sequential(
            nn.Linear(self.sbj_in_size, self.sbj_in_size),
            nn.ReLU(True),
        )
        self.linear_channel_decrease2 = nn.Sequential(
            nn.Linear(self.sbj_in_size, self.sbj_in_size),
            nn.ReLU(True),
        )

        # Gate Network
        self.auxiliarynet = AuxiliaryNet(self.aux_in_size, self.aux_out_size, self.tau)
        
        self.gate_softmax = nn.Softmax(dim=1)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, linear_h, repeat_num=1000, istrain='train'):
        # multi-dimensional aesthetic channel
        subjective_channels = {}
        subjective_channels['linear_channel_increase1'] = self.linear_channel_increase1(linear_h)  # dim: (batch_size, sbj_in_size)
        subjective_channels['linear_channel_increase2'] = self.linear_channel_increase2(linear_h)
        subjective_channels['linear_channel_default'] = self.linear_channel_default(linear_h)
        subjective_channels['linear_channel_decrease1'] = self.linear_channel_decrease1(linear_h)
        subjective_channels['linear_channel_decrease2'] = self.linear_channel_decrease2(linear_h)
        hidden_channel_h = torch.cat((subjective_channels['linear_channel_increase1'], subjective_channels['linear_channel_increase2'],
                                        subjective_channels['linear_channel_default'], subjective_channels['linear_channel_decrease1'],
                                        subjective_channels['linear_channel_decrease2']), 1)
        hidden_channel_h = hidden_channel_h.view((hidden_channel_h.size(0), self.channel_num, self.sbj_in_size))  # dim: (batch_size, channel_num, sbj_in_size)

        # gate
        if istrain == 'train' or istrain == 'val' or istrain == 'test':
            g_t, p_t = self.auxiliarynet(subjective_channels, repeat_num, istrain)  # dim: (batch_size, channel_num, repeat_num)  (batch_size, channel_num, 1)
            v_t = self.gate_softmax(g_t)
            for channel in range(self.channel_num):
                if channel == 0:
                    r_g = torch.mul(hidden_channel_h[:, channel, :].unsqueeze(-1), v_t[:, channel, :].unsqueeze(1))
                else:
                    r_g += torch.mul(hidden_channel_h[:, channel, :].unsqueeze(-1), v_t[:, channel, :].unsqueeze(1))
        
        return r_g, g_t, p_t


class AuxiliaryNet(nn.Module):
    """
    Fully Connection Layer & Gate Network

    Args:
        aux_in_size: input vector size for AuxiliaryNet
        aux_out_size: output vector size for AuxiliaryNet
        sample_num: number for sample Gate value
        tau: temperature for Gunbel-Softmax
    """
    def __init__(self, aux_in_size, aux_out_size, tau=0.1):
        super(AuxiliaryNet, self).__init__()

        self.aux_in_size = aux_in_size
        self.aux_out_size = aux_out_size
        self.tau = tau

        # Multi-Gate
        # self.gate = nn.Sequential(
        #     nn.Linear(self.aux_in_size, aux_out_size),
        #     nn.Sigmoid(),
        # )
        self.gate_increase1 = nn.Sequential(
            nn.Linear(self.aux_in_size, aux_out_size),
            nn.Sigmoid(),
        )
        self.gate_increase2 = nn.Sequential(
            nn.Linear(self.aux_in_size, aux_out_size),
            nn.Sigmoid(),
        )
        self.gate_default = nn.Sequential(
            nn.Linear(self.aux_in_size, aux_out_size),
            nn.Sigmoid(),
        )
        self.gate_decrease1 = nn.Sequential(
            nn.Linear(self.aux_in_size, aux_out_size),
            nn.Sigmoid(),
        )
        self.gate_decrease2 = nn.Sequential(
            nn.Linear(self.aux_in_size, aux_out_size),
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

    def forward(self, linear_h, repeat_num=1000, istrain='train'):
        p_t_increase1 = self.gate_increase1(linear_h['linear_channel_increase1'])  # dim: (batch_size, 1)
        p_t_increase2 = self.gate_increase2(linear_h['linear_channel_increase2'])
        p_t_default = self.gate_default(linear_h['linear_channel_default'])
        p_t_decrease1 = self.gate_decrease1(linear_h['linear_channel_decrease1'])
        p_t_decrease2 = self.gate_decrease2(linear_h['linear_channel_decrease2'])
        p_t_ = torch.cat((p_t_increase1, p_t_increase2, p_t_default, p_t_decrease1, p_t_decrease2), 1).unsqueeze(-1)  # dim: (batch_size, channel_num, 1)
        g_t = torch.zeros(p_t_.size(0), p_t_.size(1), repeat_num).cuda()  # dim: (batch_size, 1, repeat_num)

        if istrain == 'train':
            p_t = p_t_.repeat(1, 1, 2)  # dim: (batch_size, 1, 2)
            p_t[:, :, 0] = 1 - p_t[:, :, 0]
            for time in range(repeat_num):
                g_hat = F.gumbel_softmax(p_t, self.tau, hard=False)
                g_t[:, :, time] = g_hat[:, :, 1]
        elif istrain == 'val' or istrain == 'test':
            m = torch.distributions.bernoulli.Bernoulli(p_t_)
            for time in range(repeat_num):
                g_t[:, :, time] = m.sample().squeeze()

        return g_t, p_t_  # dim: (batch_size, 5, repeat_num)  (batch_size, 5, 1)


def LUPVisQ_loss(logits, target, g_t, p_t, repeat_num=1000, lambda_=1e-3):
    """
    Args:
        logits:  dim: repeat_num x (batch_size, class_num)
        target:  dim: (batch_size)
        g_t:     dim: (batch_size, channel_num, repeat_num)
        p_t:     dim: (batch_size, channel_num, 1)
    """
    p_targets = np.zeros((g_t.size(0), repeat_num), dtype=float)     # dim: (batch_size, repeat_num)
    target_list = target.cpu().tolist()
    logits_ = torch.zeros((logits[0].size(0), logits[0].size(1)), dtype=torch.float).cuda()
    g_t_ = torch.zeros((g_t.size(0), g_t.size(1), 1), dtype=torch.float).cuda()

    for time in range(repeat_num):
        p_logits_time = torch.zeros((g_t.size(0), 1), dtype=torch.float).cuda()  # dim: (batch_size, 1)
        for batch_index, logit in enumerate(logits[time]):           # dim: (batch_size, class_num)
            p_logits_time[batch_index, :] = logit[target_list[batch_index]]
        
        g_t_time = g_t[:, :, time].unsqueeze(-1)  # dim: (batch_size, channel_num, 1)
        p_t_time = p_t                            # dim: (batch_size, channel_num, 1)
        p_g_t = torch.sum((p_t_time ** g_t_time * (1-p_t_time) ** (1-g_t_time)), 1)  # dim: (batch_size, 1)
        
        p_targets[:, time] = (p_logits_time * p_g_t).flatten().detach().cpu().numpy()

    p_max = np.argmax(p_targets, axis=1).tolist()  # dim: (batch_size)
    for batch_index, max_index in enumerate(p_max):
        logits_[batch_index, :] = logits[max_index][batch_index, :]  # dim: (batch_size, class_num)
        g_t_[batch_index, :, :] = g_t[batch_index, :, max_index].unsqueeze(-1)  # dim: (batch_size, channel_num, 1)
    
    T = g_t.size(1)
    loss = F.cross_entropy(logits_, target)
    regular = (lambda_ * torch.sum(g_t))/T
    return loss, logits_
