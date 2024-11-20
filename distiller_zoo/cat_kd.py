import torch
import torch.nn as nn
import torch.nn.functional as F
    
class CAT_KD(nn.Module):

    def __init__(self, cam_resolution = 2, normalize = False, binarize = False, onlyTransferPartialCAMs = False, cams_nums = 100, kd_strategy=0):
        super(CAT_KD, self).__init__()
        self.CAM_RESOLUTION = cam_resolution
        self.relu = nn.ReLU()

        self.IF_NORMALIZE = normalize
        self.IF_BINARIZE = binarize

        self.IF_OnlyTransferPartialCAMs = onlyTransferPartialCAMs
        self.CAMs_Nums = cams_nums
        # 0: select CAMs with top x predicted classes
        # 1: select CAMs with the lowest x predicted classes
        self.Strategy = kd_strategy
    
    def forward(self, f_s, f_t, l_t):
        # perform binarization
        if self.IF_BINARIZE:
            n,c,h,w = f_t.shape
            threshold = torch.norm(f_t, dim=(2,3), keepdim=True, p=1)/(h*w)
            f_t = f_t - threshold
            f_t = self.relu(f_t).bool() * torch.ones_like(f_t)
        
        # only transfer CAMs of certain classes
        if self.IF_OnlyTransferPartialCAMs:
            n,c,w,h = f_t.shape
            with torch.no_grad():
                if self.Strategy==0:
                    l = torch.sort(l_t, descending=True)[0][:, self.CAMs_Nums-1].view(n,1)
                    mask = self.relu(l_t-l).bool()
                    mask = mask.unsqueeze(-1).reshape(n,c,1,1)
                elif self.Strategy==1:
                    l = torch.sort(l_t, descending=True)[0][:, 99-self.CAMs_Nums].view(n,1)
                    mask = self.relu(l_t-l).bool()
                    mask = ~mask.unsqueeze(-1).reshape(n,c,1,1)
            f_t,f_s = _mask(f_t,f_s,mask)

        loss_feat = CAT_loss(f_s, f_t, self.CAM_RESOLUTION, self.IF_NORMALIZE)
        
        return loss_feat

def _Normalize(feat,IF_NORMALIZE):
    if IF_NORMALIZE:
        feat = F.normalize(feat,dim=(2,3))
    return feat

def CAT_loss(CAM_Student, CAM_Teacher, CAM_RESOLUTION, IF_NORMALIZE):   
    CAM_Student = F.adaptive_avg_pool2d(CAM_Student, (CAM_RESOLUTION, CAM_RESOLUTION))
    CAM_Teacher = F.adaptive_avg_pool2d(CAM_Teacher, (CAM_RESOLUTION, CAM_RESOLUTION))
    loss = F.mse_loss(_Normalize(CAM_Student, IF_NORMALIZE), _Normalize(CAM_Teacher, IF_NORMALIZE))
    return loss
    

def _mask(tea,stu,mask):
    n,c,w,h = tea.shape
    mid = torch.ones(n,c,w,h).cuda()
    mask_temp = mask.view(n,c,1,1)*mid.bool()
    t=torch.masked_select(tea, mask_temp)
    
    if (len(t))%(n*w*h)!=0:
        return tea, stu

    n,c,w_stu,h_stu = stu.shape
    mid = torch.ones(n,c,w_stu,h_stu).cuda()
    mask = mask.view(n,c,1,1)*mid.bool()
    stu=torch.masked_select(stu, mask)
    
    return t.view(n,-1,w,h), stu.view(n,-1,w_stu,h_stu)