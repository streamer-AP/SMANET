import torch
import torch. nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Gaussian(nn.Module):
    def __init__(self, in_channels, sigmalist, kernel_size=64, stride=1, padding=0, froze=True):
        super(Gaussian, self).__init__()
        out_channels = len(sigmalist) * in_channels

        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma: math.exp(-(x - mu) ** 2 / float(2 * sigma ** 2)) if sigma > 0 else 0.0)
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            if sigma <= 0:
                print(f"[ERROR] Gaussian sigma={sigma} <= 0, setting to 1.0")
                sigma = 1.0
            gauss = torch.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss_sum = gauss.sum()
            if gauss_sum == 0 or not torch.isfinite(gauss_sum):
                print(f"[ERROR] Gaussian kernel sum is {gauss_sum}, using uniform kernel")
                gauss = torch.ones_like(gauss) / gauss.numel()
            else:
                gauss /= gauss_sum

            if not torch.isfinite(gauss).all():
                print(f"[ERROR] Gaussian kernel contains NaN/Inf after normalization, replacing with uniform")
                gauss = torch.ones_like(gauss) / gauss.numel()
            _1D_window = gauss.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(in_channels, 1, kernel_size, kernel_size).contiguous())
            windows.append(window)
        kernels = torch.stack(windows)
        kernels = kernels.permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)

        if not torch.isfinite(weight).all():
            print(f"[ERROR] Gaussian weight contains NaN/Inf, replacing with uniform kernel")
            weight = torch.ones_like(weight) / weight[0, 0].numel()

        self.gkernel = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.gkernel.weight = torch.nn.Parameter(weight)

        if froze: self.frozePara()

    def forward(self, dotmaps):

        if not torch.isfinite(dotmaps).all():
            print(f"[WARN] Gaussian input contains NaN/Inf, replacing")
            dotmaps = torch.nan_to_num(dotmaps, nan=0.0, posinf=1.0, neginf=0.0)

        gaussianmaps = self.gkernel(dotmaps)

        if not torch.isfinite(gaussianmaps).all():
            print(f"[ERROR] Gaussian output contains NaN/Inf, replacing")
            gaussianmaps = torch.nan_to_num(gaussianmaps, nan=0.0, posinf=1.0, neginf=0.0)

        return gaussianmaps

    def frozePara(self):
        for para in self.parameters():
            para.requires_grad = False

class SumPool2d(nn.Module):
    def __init__(self, kernel_size):
        super(SumPool2d, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        if type(kernel_size) is not int:
            self.area = kernel_size[0] * kernel_size[1]
        else:
            self.area = kernel_size * self.kernel_size

    def forward(self, dotmap):
        return self.avgpool(dotmap) * self.area
