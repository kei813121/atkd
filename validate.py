"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import time
import numpy as np
import random

import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed, Global_T, CosineDecay, LinearDecay
from models.util import Connector, Translator, Paraphraser
from dataset import get_dataset, get_dataset_strong
from helper.util import adjust_learning_rate
from distiller_zoo import TTM, WTTM, DistillKL, CRDLoss, ITLoss, HintLoss, Attention, RKDLoss, OFD, ReviewKD, SimKD, CAT_KD, MLKD, DKD, DIST, APKD
from helper.loops import train_distill as train, validate


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')
    parser.add_argument('--model_path', type=str, default=None, help='model snapshot')
    parser.add_argument('--seed', type=int, default=0, help='seed id, set to 0 if do not want to fix the seed')
    parser.add_argument('--distill', type=str, default=None, help='dummy for dataloader')

    opt = parser.parse_args()

    opt.model = get_teacher_name(opt.model_path)
    
    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    if opt.seed:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # dataloader
    train_loader, val_loader, n_data, n_cls = get_dataset(opt)

    # model
    model_t = load_teacher(opt.model_path, n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    feat_t, _ = model_t(data, is_feat=True)

    module_list = nn.ModuleList([])

    criterion_cls = nn.CrossEntropyLoss()

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_cls.cuda()
        if not opt.seed:
            cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('accuracy: ', teacher_acc)

if __name__ == '__main__':
    main()
