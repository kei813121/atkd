import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from helper.util import normalize, calc_maxlogit_temp

def kd_loss(logits_student_in, logits_teacher_in, temperature, reduce=True, logit_stand=False):
    # logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    # logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    logits_student = logits_student_in
    logits_teacher = logits_teacher_in
    T = temperature

    log_pred_student = F.log_softmax(logits_student / T, dim=1)
    pred_teacher = F.softmax(logits_teacher / T, dim=1)
    # if reduce:
    #     loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    # else:
    #     loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    # loss_kd *= T**2
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none") * (T**2)
    if reduce:
        loss_kd = loss_kd.sum(1).mean()
    else:
        loss_kd = loss_kd.sum(1)
    return loss_kd

def cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss

def bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_conf(x, y, lam, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = lam.reshape(-1,1,1,1)
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class MLKD(nn.Module):
    def __init__(self, T, logit_stand=False, mlogit_temp=False):
        super(MLKD, self).__init__()
        self.T = T
        self.logit_stand = logit_stand
        self.mlogit_temp = mlogit_temp
    
    def forward(self, y_s, y_t):
        logits_student_w = y_s[0]
        logits_teacher_w = y_t[0]
        logits_student_s = y_s[1]
        logits_teacher_s = y_t[1]
        
        logits_student_weak = normalize(logits_student_w) if self.logit_stand else logits_student_w
        logits_teacher_weak = normalize(logits_teacher_w) if self.logit_stand else logits_teacher_w
        logits_student_strong = normalize(logits_student_s) if self.logit_stand else logits_student_s
        logits_teacher_strong = normalize(logits_teacher_s) if self.logit_stand else logits_teacher_s
        
        temperature_weak = calc_maxlogit_temp(logits_student_weak, logits_teacher_weak) if self.logit_stand and self.mlogit_temp else self.T
        temperature_strong = calc_maxlogit_temp(logits_student_strong, logits_teacher_strong) if self.logit_stand and self.mlogit_temp else self.T

        batch_size, class_num = logits_student_strong.shape

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        loss_kd_weak = ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            # self.T,
            temperature_weak,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + ((kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean())

        loss_kd_strong = kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            # self.T,
            temperature_strong,
            logit_stand=self.logit_stand,
        ) + kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
            logit_stand=self.logit_stand,
        ) + kd_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
            logit_stand=self.logit_stand,
        ) + kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
            logit_stand=self.logit_stand,
        ) + kd_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
            logit_stand=self.logit_stand,
        )

        loss_cc_weak = ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            # self.T,
            temperature_weak,
            # reduce=False
        ) * class_conf_mask).mean()) + ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * class_conf_mask).mean()) + ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * class_conf_mask).mean()) + ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * class_conf_mask).mean()) + ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * class_conf_mask).mean())
        loss_cc_strong = cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            # self.T,
            temperature_strong,
        ) + cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) + cc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) + cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) + cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        )
        loss_bc_weak = ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            # self.T,
            temperature_weak,
        ) * mask).mean()) + ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            3.0,
        ) * mask).mean()) + ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            5.0,
        ) * mask).mean()) + ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            2.0,
        ) * mask).mean()) + ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            6.0,
        ) * mask).mean())
        loss_bc_strong = ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            # self.T,
            temperature_strong,
        ) * mask).mean()) + ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            3.0,
        ) * mask).mean()) + ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            5.0,
        ) * mask).mean()) + ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            2.0,
        ) * mask).mean()) + ((bc_loss(
            logits_student_strong,
            logits_teacher_strong,
            6.0,
        ) * mask).mean())

        losses_dict = {
            "loss_kd": loss_kd_weak + loss_kd_strong,
            "loss_cc": loss_cc_weak,
            "loss_bc": loss_bc_weak
        }

        return sum([l.mean() for l in losses_dict.values()])
