import math
from pathlib import Path
from collections import defaultdict
import glob
import re
import random

import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(list)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].append(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        meter_str = []
        for name, meter in self.meters.items():
            meter_str.append(
                "{}: {}".format(name, str(meter[-1]))
            )
        return self.delimiter.join(meter_str)

    def output_csv(self, path):
        df = pd.DataFrame(self.meters)
        df.to_csv(path)

    def plot(self, path, nums_crow=1):
        nums_params = len(self.meters)
        nums_line = nums_params // nums_crow + int(nums_params % nums_crow != 0)
        plt.figure(figsize=(16, 8))
        index = 1
        for key, value in self.meters.items():
            plt.subplot(nums_line, nums_crow, index)
            plt.plot(value)
            plt.title(key)
            index += 1
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.savefig(path)


def init_seeds(seed=1024, strict=False):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if strict:
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        cudnn.benchmark, cudnn.deterministic = True, False


def warmup_cosine_schedule(optimizer, warmup_steps, total_steps):
    def f(x):
        if x < warmup_steps:
            return (float(x) + 1) / warmup_steps
        else:
            return 0.5 + 0.5 * math.cos((x - warmup_steps) / (total_steps - warmup_steps) * math.pi)

    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=f)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, gamma=1.5, alpha=0.25, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, pred, true):
        pred_prob = torch.sigmoid(pred)  # prob from logits
        loss = nn.BCEWithLogitsLoss(reduction='none')(pred, true)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # ????????????????????????????????????????????????????????????
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.pos_weight is not None:
            loss *= self.pos_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def mixup_data(x, y, mixup_alpha=1., cutmix_alpha=0., switch_prob=0.5,):
    use_cutmix = False
    if mixup_alpha > 0. and cutmix_alpha > 0.:
        use_cutmix = np.random.rand() < switch_prob
        lam = np.random.beta(cutmix_alpha, cutmix_alpha) if use_cutmix else np.random.beta(mixup_alpha, mixup_alpha)
    elif mixup_alpha > 0.:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
    elif cutmix_alpha > 0.:
        use_cutmix = True
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    else:
        lam = 1

    device = x.device
    bs, c, w, h = x.shape
    index = torch.randperm(bs).to(device)

    if use_cutmix:
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    else:
        x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def one_hot(x, num_classes, on_value=1., off_value=0.):
    x = x.long().view(-1, 1)
    device = x.device
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=1000, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        x = x.float()
        labels = labels.float()
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

if __name__ == '__main__':
    y = torch.ones(32)
    index = torch.randperm(32)
    print(y[index])
