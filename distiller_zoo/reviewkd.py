import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb

def hcl_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

def build_review_kd(student_model, teacher_model):
    out_shapes, in_channels, out_channels, shapes = None, None, None, None
    if 'x4' in student_model:
        in_channels = [64,128,256,256]
        out_channels = [64,128,256,256]
        if 'wrn' in teacher_model:
            out_channels = [32,64,128,128]
        shapes = [1,8,16,32]
    elif 'ResNet50' in student_model:
        in_channels = [16,32,64,64]
        out_channels = [16,32,64,64]
        shapes = [1,8,16,32,32]
        assert False
    elif 'resnet' in student_model:
        in_channels = [16,32,64,64]
        out_channels = [16,32,64,64]
        shapes = [1,8,16,32,32]
    elif 'vgg' in student_model:
        in_channels = [128,256,512,512,512]
        shapes = [1,4,4,8,16]
        if 'ResNet50' in teacher_model:
            out_channels = [256,512,1024,2048,2048]
            out_shapes = [1,4,8,16,32]
        else:
            out_channels = [128,256,512,512,512]
    elif 'Mobile' in student_model:
        in_channels = [12,16,48,160,1280]
        shapes = [1,2,4,8,16]
        if 'ResNet50' in teacher_model:
            out_channels = [256,512,1024,2048,2048]
            out_shapes = [1,4,8,16,32]
        elif 'wrn' in teacher_model:
            out_channels = [32,64,128,128,256]
            out_shapes = [1,1,8,16,32]
        else:
            out_channels = [128,256,512,512,512]
            out_shapes = [1,4,4,8,16]
    elif 'ShuffleV1' in student_model:
        in_channels = [240,480,960,960]
        shapes = [1,4,8,16]
        if 'wrn' in teacher_model:
            out_channels = [32,64,128,128]
            out_shapes = [1,8,16,32]
        else:
            out_channels = [64,128,256,256]
            out_shapes = [1,8,16,32]
    elif 'ShuffleV2' in student_model:
        in_channels = [116,232,464,1024]
        shapes = [1,4,8,16]
        out_channels = [64,128,256,256]
        out_shapes = [1,8,16,32]
    elif 'wrn' in student_model:
        r=int(student_model[-1:])
        in_channels = [16*r,32*r,64*r,64*r]
        out_channels = [32,64,128,128]
        if 'x4' in teacher_model:
            out_channels = [64,128,256,256]
        shapes = [1,8,16,32]
    else:
        assert False

    return shapes, out_shapes, in_channels, out_channels


class ReviewKD(nn.Module):
    def __init__(self, student_model, teacher_model, warmup_epochs=20, max_mid_channel=512):
        super(ReviewKD, self).__init__()
        self.shapes, self.out_shapes, self.in_channels, self.out_channels = build_review_kd(student_model, teacher_model)
        self.out_shapes = self.shapes if self.out_shapes is None else self.out_shapes
        
        self.warmup_epochs = warmup_epochs
        self.max_mid_channel = max_mid_channel

        abfs = nn.ModuleList()
        mid_channel = min(512, self.in_channels[-1])
        for idx, in_channel in enumerate(self.in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    self.out_channels[idx],
                    idx < len(self.in_channels) - 1,
                )
            )
        self.abfs = abfs[::-1]
    
    def forward(self, f_s, f_t, **kwargs):
        x = f_s[:-1] + [f_s[-1].unsqueeze(-1).unsqueeze(-1)]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        f_t = f_t[1:-1] + [f_t[-1].unsqueeze(-1).unsqueeze(-1)]
        # losses
        loss_reviewkd = (min(kwargs["epoch"] / self.warmup_epochs, 1.0) * hcl_loss(results, f_t))
        return loss_reviewkd


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x