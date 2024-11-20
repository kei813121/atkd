from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from helper.util import normalize, calc_maxlogit_temp

class DistillKL(nn.Module):
    def __init__(self, T, logit_stand=False, mlogit_temp=False):
        super(DistillKL, self).__init__()
        self.T = T
        self.logit_stand = logit_stand
        self.mlogit_temp = mlogit_temp

    def forward(self, y_s, y_t):
        l_s = normalize(y_s) if self.logit_stand else y_s
        l_t = normalize(y_t) if self.logit_stand else y_t

        self.T = calc_maxlogit_temp(l_s, l_t) if self.logit_stand and self.mlogit_temp else self.T

        p_s = F.log_softmax(l_s/self.T, dim=1)
        p_t = F.softmax(l_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='none') * (self.T ** 2)
        loss = loss.sum() / y_s.shape[0]

        return loss