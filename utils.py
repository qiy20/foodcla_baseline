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
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # 反而形成了对正样本的抑制，这很奇怪呀！！
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


if __name__ == '__main__':
    y = torch.ones(32)
    index = torch.randperm(32)
    print(y[index])
