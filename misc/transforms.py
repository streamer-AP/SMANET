import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from config import cfg
import torch
import torch.nn.functional as F
import numpy
import pdb
import cv2
from torchvision.transforms import functional as TrF
from misc import inflation


def resize_flow(flow, size, scale_x, scale_y):
    if flow is None:
        return None

    flow = F.interpolate(
        flow.unsqueeze(0).float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    flow[0] *= scale_x
    flow[1] *= scale_y
    return flow


def flow_to_image_size(flow, image_size, reference_size):
    image_h, image_w = image_size
    reference_h, reference_w = reference_size
    return resize_flow(
        flow,
        (image_h, image_w),
        image_w / reference_w,
        image_h / reference_h,
    )

class ProcessSub(object):
    def __init__(self, T=0.1, K=51):
        self.T = T
        self.inf = inflation.inflation(K=K)

    def getHS(self, flow):

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        h = ang * 180 / np.pi / 2
        s = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return h, s

    def __call__(self, flow):
        h, s = self.getHS(flow[:, :, 0:2])
        flow[:, :, 0] = h.astype(np.float32) / 255
        flow[:, :, 1] = s.astype(np.float32) / 255

        temp = np.ones(flow[:, :, 2].shape)
        temp[abs(flow[:, :, 2]) < self.T] = 0
        flow[:, :, 2] = flow[:, :, 2] * temp

        return flow

class RandomEmptyFlow(object):
    def __call__(self, flow):
        if random.random() < 0.04:
            flow = numpy.zeros((flow.shape[0], flow.shape[1], flow.shape[2])).astype(numpy.float32)
        return flow

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class RandomHorizontallyFlip(object):
    def __init__(self, task=None):
        self.task = task

    def __call__(self, img, gt, flip_flag=0, bbx=None, flow=None):

        if flip_flag :
            w, h = img.size
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt['points'][:, 0] = w - 1 - gt['points'][:, 0]
            if flow is not None:
                flow = flow.flip(-1).clone()
                flow[0].neg_()

        if flow is None:
            return img, gt
        return img, gt, flow

class RandomVerticallyFlip(object):
    def __call__(self, img, mask, flow=None, bbx=None):
        if random.random() < 0.5:
            if bbx is None:

                return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
            w, h = img.size
            ymin = w - bbx[:, 2]
            ymax = w - bbx[:, 0]
            bbx[:, 0] = ymin
            bbx[:, 2] = ymax
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM), bbx
        if bbx is None:
            return img, mask
        return img, mask, bbx

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size),)
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))

class ScaleByRateWithMin(object):
    def __init__(self, min_w, min_h, task=None):
        self.min_w = min_w
        self.min_h = min_h
        self.task = task

    def __call__(self, img, gt, flow=None):
        w, h = img.size

        new_w = self.min_w
        new_h = self.min_h
        img = img.resize((new_w, new_h), Image.LANCZOS)

        rate = new_w / w
        gt['points'] =  gt['points']  * rate
        if flow is None:
            return img, gt
        flow = resize_flow(flow, (new_h, new_w), new_w / w, new_h / h)
        return img, gt, flow

class Scale(object):
    def __init__(self, min_w, min_h):
        self.min_w = min_w
        self.min_h = min_h

    def __call__(self, img, gt):
        w, h = img.size
        new_w = self.min_w
        new_h = self.min_h
        img = img.resize((new_w, new_h), Image.LANCZOS)
        rate_w = new_w / w
        rate_h = new_h / h
        gt['points'][:, 0] =  gt['points'][:, 0]  * rate_w
        gt['points'][:, 1] =  gt['points'][:, 1]  * rate_h
        return img, gt

def check_image(img, target, crop_size, max_size, flow=None):
    w, h = img.size
    long_side = max(w, h)
    short_side = min(w, h)
    max_long_side, max_short_side = max_size
    scale_long = max_long_side / long_side
    scale_short = max_short_side / short_side
    if scale_long < 1 or scale_short < 1:
        scale = min(scale_long, scale_short)
        new_width = int(w * scale)
        new_height = int(h * scale)
        target['points'] = target['points'] * scale
        img = img.resize((new_width, new_height), Image.LANCZOS)
        if flow is not None:
            flow = resize_flow(flow, (new_height, new_width), scale, scale)
    c_h, c_w = crop_size
    w, h = img.size
    if w < c_w or h < c_h:
        delta_w = max(c_w - w, 0)
        delta_h = max(c_h - h, 0)
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)
        target['points'] = target['points']+torch.tensor([delta_w // 2, delta_h // 2], dtype = torch.float32)
        if flow is not None:
            flow = F.pad(flow, (padding[0], padding[2], padding[1], padding[3]))

    if flow is None:
        return img, target
    return img, target, flow

class RandomCrop(object):
    def __init__(self):
        pass

    def __call__(self, img, gt, crop_left, crop_size, flow=None):

        th, tw = crop_size[0], crop_size[1]
        x1, y1 = crop_left
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        if flow is not None:
            flow = flow[:, y1:y1 + th, x1:x1 + tw]
        index = (gt['points'][:,0]>x1+1) & (gt['points'][:,0]<x1 + tw-1) & (gt['points'][:,1]>y1+1) & (gt['points'][:,1]<y1 + th-1)

        gt['points'] = gt['points'][index].view(-1,2).contiguous()
        gt['points'] -= torch.tensor([x1, y1], dtype = torch.float32)

        if 'person_id' in gt:
            gt['person_id'] =  gt['person_id'][index]
        elif 'inflow' in gt:
            gt['inflow'] = gt['inflow'][index]
            gt['outflow'] = gt['outflow'][index]
        else:
            raise("error!")
        if flow is None:
            return img, gt
        return img, gt, flow

class FreeScale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]), Image.NEAREST)

class ScaleDown(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, mask):
        return mask.resize((self.size[1] / cfg.TRAIN.DOWNRATE, self.size[0] / cfg.TRAIN.DOWNRATE), Image.NEAREST)

class RGB2Gray(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, img):
        if random.random() < 0.1:
            return TrF.to_grayscale(img, num_output_channels=3)
        else:
            return img

class GammaCorrection(object):
    def __init__(self, gamma_range=[0.4, 2]):
        self.gamma_range = gamma_range

    def __call__(self, img):
        if random.random() < 0.5:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            return TrF.adjust_gamma(img, gamma)
        else:
            return img

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):

        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor * self.para
        return tensor

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor == 1:
            return img
        tmp = np.array(img.resize((w // self.factor, h // self.factor), Image.BICUBIC)) * self.factor * self.factor
        img = Image.fromarray(tmp)
        return img

class tensormul(object):
    def __init__(self, mu=255.0):
        self.mu = 255.0

    def __call__(self, _tensor):
        _tensor.mul_(self.mu)
        return _tensor
