import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
import math

def feat_loss(source, target, margin):
    margin = margin.to(source)
    loss = (
        (source - margin) ** 2 * ((source > margin) & (target <= margin)).float()
        + (source - target) ** 2
        * ((source > target) & (target > margin) & (target <= 0)).float()
        + (source - target) ** 2 * (target > 0).float()
    )
    return torch.abs(loss).mean(dim=0).sum()

class ConnectorConvBN(nn.Module):
    def __init__(self, s_channels, t_channels, kernel_size=1):
        super(ConnectorConvBN, self).__init__()
        self.s_channels = s_channels
        self.t_channels = t_channels
        self.connectors = nn.ModuleList(
            self._make_conenctors(s_channels, t_channels, kernel_size)
        )

    def _make_conenctors(self, s_channels, t_channels, kernel_size):
        assert len(s_channels) == len(t_channels), "unequal length of feat list"
        connectors = nn.ModuleList(
            [
                self._build_feature_connector(t, s, kernel_size)
                for t, s in zip(t_channels, s_channels)
            ]
        )
        return connectors

    def _build_feature_connector(self, t_channel, s_channel, kernel_size):
        C = [
            nn.Conv2d(
                s_channel,
                t_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(t_channel),
        ]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out

class OFD(nn.Module):
    def __init__(self, teacher_stage_channels, student_stage_channels, bn_before_relu, kernel_size=1):
        super(OFD, self).__init__()
        self.init_ofd_modules(
            tea_channels = teacher_stage_channels,
            stu_channels = student_stage_channels,
            bn_before_relu = bn_before_relu,
            kernel_size = kernel_size)

    def init_ofd_modules(self, tea_channels, stu_channels, bn_before_relu, kernel_size):
        tea_channels, stu_channels = self._align_list(tea_channels, stu_channels)
        self.connectors = ConnectorConvBN(stu_channels, tea_channels, kernel_size=kernel_size)

        self.margins = []
        for idx, bn in enumerate(bn_before_relu):
            margin = []
            std = bn.weight.data
            mean = bn.bias.data
            for (s, m) in zip(std, mean):
                s = abs(s.item())
                m = m.item()
                if norm.cdf(-m / s) > 0.001:
                    margin.append(
                        -s
                        * math.exp(-((m / s) ** 2) / 2)
                        / math.sqrt(2 * math.pi)
                        / norm.cdf(-m / s)
                        + m
                    )
                else:
                    margin.append(-3 * s)
            margin = torch.FloatTensor(margin).to(std.device)
            self.margins.append(margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())


    def ofd_loss(self, feature_student, feature_teacher):
        feature_student, feature_teacher = self._align_list(
            feature_student, feature_teacher
        )
        feature_student = [
            self.connectors.connectors[idx](feat)
            for idx, feat in enumerate(feature_student)
        ]

        loss_distill = 0
        feat_num = len(feature_student)
        for i in range(feat_num):
            loss_distill = loss_distill + feat_loss(
                feature_student[i],
                F.adaptive_avg_pool2d(
                    feature_teacher[i], feature_student[i].shape[-2:]
                ).detach(),
                self.margins[i],
            ) / 2 ** (feat_num - i - 1)
        return loss_distill

    def forward(self, f_s, f_t):
        # losses
        # loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # loss_feat = self.ofd_loss(
        #     f_s["preact_feats"][1:], f_t["preact_feats"][1:]
        # )
        loss_feat = self.ofd_loss(f_s, f_t)
        return loss_feat
        # losses_dict = {"loss_ce": loss_ce, "loss_kd": loss_feat}
        # return logits_student, losses_dict

    def _align_list(self, *input_list):
        min_len = min([len(l) for l in input_list])
        return [l[-min_len:] for l in input_list]
