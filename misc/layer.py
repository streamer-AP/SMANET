import torch
import torch.nn as nn

from .dot_ops import Gaussian, SumPool2d
import scipy.spatial
import scipy.ndimage
import numpy as np
import  torch.nn.functional as F
import cv2 as cv
class Point2Mask(object):
    def __init__(self,  max_kernel_size=7):

        self.max_kernel_size = max_kernel_size
    def __call__(self, target, pre_map):
        b,c,h,w = pre_map.size()
        mask_map = torch.zeros_like(pre_map)
        for idx, sub_target in enumerate(target):
            points = sub_target["points"]

            count = points.shape[0]
            if count==0:
                continue
            elif count==1:
                pt = points[0].astype(np.int32)
                kernel_size = self.max_kernel_size
                up = max(pt[1] - kernel_size, 0)
                down = min(pt[1] + kernel_size + 1, h)
                left = max(pt[0] - kernel_size, 0)
                right = min(pt[0] + kernel_size + 1, w)

                mask_map[idx, 0, up:down + 1, left:right + 1] = 1
            else:
                leafsize = 2048
                tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
                distances, locations = tree.query(points, k=2)
                for i, pt in enumerate(points):
                    if pt[0] >= w or pt[1] > h:
                        continue
                    pt = pt.astype(np.int32)
                    kernel_size = (distances[i][1]) * 0.25
                    kernel_size = min(self.max_kernel_size, int(kernel_size + 0.5))
                    up = max(pt[1] - kernel_size,0)
                    down = min(pt[1] + kernel_size+1,h)
                    left = max(pt[0] - kernel_size,0)
                    right = min(pt[0] + kernel_size+1,w)
                    mask_map[idx,0, up:down+1, left:right+1]=1

        return  mask_map
class Gaussianlayer(nn.Module):
    def __init__(self, sigma=None, kernel_size=15):
        super(Gaussianlayer, self).__init__()
        if sigma == None:
            sigma = [4]
        self.gaussian = Gaussian(1, sigma, kernel_size=kernel_size, padding=kernel_size//2, froze=True)

    def forward(self, dotmaps):
        denmaps = self.gaussian(dotmaps)
        return denmaps

class PointsToHeatmap(nn.Module):
    def __init__(self, sigma=4, kernel_size=31):
        super(PointsToHeatmap, self).__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        kernel_range = torch.arange(self.kernel_size, dtype=torch.float32) - self.padding
        x, y = torch.meshgrid(kernel_range, kernel_range, indexing='ij')
        gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))

        self.register_buffer('gaussian_kernel', gaussian_kernel)

    def forward(self, B, C, H, W, points_list):
        heatmap = torch.zeros((B, C, H, W), device=self.gaussian_kernel.device)

        for b_idx in range(B):
            points = points_list[b_idx]
            if points.numel() == 0:
                continue

            for pt in points:
                x, y = int(pt[0]), int(pt[1])

                y_start, y_end = max(0, y - self.padding), min(H, y + self.padding + 1)
                x_start, x_end = max(0, x - self.padding), min(W, x + self.padding + 1)

                kernel_y_start = max(0, self.padding - y)
                kernel_y_end = self.kernel_size - max(0, (y + self.padding + 1) - H)
                kernel_x_start = max(0, self.padding - x)
                kernel_x_end = self.kernel_size - max(0, (x + self.padding + 1) - W)

                if y_start < y_end and x_start < x_end:

                    heatmap[b_idx, 0, y_start:y_end, x_start:x_end] = torch.maximum(
                        heatmap[b_idx, 0, y_start:y_end, x_start:x_end],
                        self.gaussian_kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
                    )

        return heatmap

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor(self.pos_weight, device=pred.device))

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.5, smooth=1e-5):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        focal_tversky = (1 - tversky)**self.gamma

        return focal_tversky

class L1HeatmapLoss(nn.Module):
    def __init__(self):
        super(L1HeatmapLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        return self.loss(pred, target)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class convDU(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(9,1)
        ):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).resize(n,c,1,w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)

        for i in range(h):
            pos = h-i-1
            if pos == h-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]

        fea = torch.cat(fea_stack, 2)
        return fea

class convLR(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(1,9)
        ):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).resize(n,c,h,1)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)

        for i in range(w):
            pos = w-i-1
            if pos == w-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]

        fea = torch.cat(fea_stack, 3)
        return fea
