from __future__ import print_function, division

import sys
import time
import torch
import torch.nn.functional as F

from .util import AverageMeter, accuracy, normalize

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    # for idx, (input, target) in enumerate(train_loader):
    for idx, data in enumerate(train_loader):
        input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if opt.have_mlp:
        decay_value = opt.gradient_decay.get_value(epoch)

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)


        # cls + kl div
        preact = False
        if opt.distill in ['mlkd']:
            input_weak, input_strong = input
            input_weak, input_strong = input_weak.float(), input_strong.float()
            if torch.cuda.is_available():
                input_weak = input_weak.cuda()
                input_strong = input_strong.cuda()
            target = target.cuda()
            index = index.cuda()
            
            # ===================forward=====================
            logit_s = torch.stack([model_s(input_weak, is_feat=True, preact=preact)[1], model_s(input_strong, is_feat=True, preact=preact)[1]])
            with torch.no_grad():
                logit_t = torch.stack([model_t(input_weak, is_feat=True, preact=preact)[1], model_t(input_strong, is_feat=True, preact=preact)[1]])

            loss_cls = criterion_cls(logit_s[0], target) + criterion_cls(logit_s[1], target)
            loss_cls = loss_cls.mean()
        else:
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()
                if opt.distill in ['crd']:
                    contrast_idx = contrast_idx.cuda()
            
            # ===================forward=====================
            if opt.distill in ['itrd', 'ofd'] or opt.stu_preact:
                preact = True
            feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                if opt.distill in ['reviewkd']:
                    feat_t, logit_t = model_t(input, is_feat=True, preact=True)
                else:
                    feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            # CTKD
            if opt.have_mlp:
                temp = opt.mlp(logit_t, logit_s, decay_value)  # (teacher_output, student_output)
                temp = opt.t_start + opt.t_end * torch.sigmoid(temp)
                temp = temp.cuda()
                if opt.distill in ['kd']:
                    criterion_kd.T = temp

            loss_cls = criterion_cls(logit_s, target)
            if opt.alpha > 0:
                loss_div = criterion_div(logit_s, logit_t)

        # cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
        cls_t = model_t.get_feat_modules()[-1]

        # other distillation loss
        if opt.distill == 'kd':
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'ofd':
            f_s = feat_s[1:-1]
            f_t = feat_t[1:-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'reviewkd':
            loss_kd = criterion_kd(feat_s, feat_t, epoch=epoch)
        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            logit_s = pred_feat_s
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)
        elif opt.distill == 'cat_kd':
            loss_kd = criterion_kd(feat_s[-2], feat_s[-2], logit_t)
        elif opt.distill == 'dkd':
            loss_kd = criterion_kd(logit_s, logit_t, target, epoch=epoch)
        elif opt.distill == 'mlkd':
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'ttm':
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'wttm':
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'itrd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_correlation = opt.lambda_corr * criterion_kd.forward_correlation_it(f_s, f_t)
            loss_mutual = opt.lambda_mutual * criterion_kd.forward_mutual_it(f_s, f_t)
            loss_kd = loss_mutual + loss_correlation
        elif opt.distill == 'dist':
            loss_kd = criterion_kd(logit_s, logit_t)
        else:
            raise NotImplementedError(opt.distill)

        if opt.distill in ['mlkd']:
            loss = opt.gamma * loss_cls + opt.beta * loss_kd
            acc1, acc5 = accuracy(logit_s[0], target, topk=(1, 5))
            losses.update(loss.item(), input[0].size(0))
            top1.update(acc1[0], input[0].size(0))
            top5.update(acc5[0], input[0].size(0))
        else:
            if opt.alpha > 0:
                loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
            else:
                loss = opt.gamma * loss_cls + opt.beta * loss_kd
            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input[0].size(0))
            top5.update(acc5[0], input.size(0))


        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt, **kwargs):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # covs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            if opt.distill == 'simkd' and 'module_list' in kwargs:
                module_list = kwargs['module_list']
                model_t = module_list[-1]
                model_t.eval()

                feat_s, _ = model(input, is_feat=True)
                feat_t, _ = model_t(input, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                # cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
                cls_t = model_t.get_feat_modules()[-1]
                _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            else:
                output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg