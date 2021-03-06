from pathlib import Path
import argparse
import time

import matplotlib.pyplot as plt
import yaml
import logging

import torch
from torch import nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from timm.models.resnet import resnet18, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d, Bottleneck, BasicBlock
from timm.models.efficientnet import efficientnet_b2
from timm.models.swin_transformer import swin_base_patch4_window7_224, swin_base_patch4_window7_224_in22k, \
    swin_base_patch4_window12_384_in22k, swin_large_patch4_window7_224_in22k, swin_small_patch4_window7_224
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy
from torch.cuda import amp
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils import init_seeds, increment_path, MetricLogger, mixup_data, one_hot, warmup_cosine_schedule, CenterLoss

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

DATA_TXT = {'train': 'train_qtcom.txt', 'val': 'val_qtcom.txt', 'test': 'test_qtcom.txt'}
DATA_DIR = {'train': 'Train_qtc', 'val': 'val', 'test': 'test_new'}


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=32, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--num_classes', default=1000, type=int)
    parser.add_argument('--lr', default=4e-5, type=float)
    parser.add_argument('--warmup_epochs', default=0, type=float)
    parser.add_argument('--label_smooth', default=0., type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--bce_loss', action='store_true')
    parser.add_argument('--center_loss', action='store_true')
    parser.add_argument('--mix_up', default=0., type=float)
    parser.add_argument('--cut_mix', default=0., type=float)
    parser.add_argument('--save_dir', default='run/exp')
    opt = parser.parse_intermixed_args()
    return opt


class FoodDataset(Dataset):
    def __init__(self, mode, infer_size=224):  # mode 'train','val' or 'test'
        self.mode = mode
        if isinstance(infer_size,int):
            infer_size = infer_size, infer_size
        self.infer_size = infer_size
        txt = DATA_TXT[mode]
        direc = Path(DATA_DIR[mode]).absolute()
        if mode in ['train', 'val']:
            img_paths, labels = [], []
            with open(txt) as f:
                for line in f.readlines():
                    path, label = line.strip('\n').split()
                    img_paths.append(str(direc / path))
                    labels.append(label)
            self.img_paths = img_paths
            self.labels = labels
        else:
            img_paths = []
            with open(txt) as f:
                for line in f.readlines():
                    path = line.strip('\n')
                    img_paths.append(str(direc / path))
            self.img_paths = img_paths
        self.trans = None
        self.trans = transforms.Compose([transforms.RandomAffine(degrees=10, translate=(0.2, 0.2),scale=(0.8, 1.5)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, ),
                                         ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = cv2.imread(self.img_paths[item])
        img=cv2.resize(img,self.infer_size)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img)) / 255.
        if self.mode == 'train':
            if self.trans:
                img = self.trans(img)
            return img, torch.tensor(int(self.labels[item])).long()
        elif self.mode == 'val':
            return img, torch.tensor(int(self.labels[item])).long()
        else:
            return img


def train(opt):
    init_seeds()
    save_dir, device, epochs, lr, weight_decay, batch_size, num_workers,img_size = Path(
        opt.save_dir), opt.device, opt.epochs, opt.lr, opt.weight_decay, opt.batch_size, opt.num_workers,opt.img_size
    # save dir
    save_dir = increment_path(save_dir)
    w = save_dir / 'weights'  # weights dir, /==.joinpath()
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    txt_res, img_res = save_dir / 'res.txt', save_dir / 'res.png'
    submit = save_dir / 'submit.csv'
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    # logger
    metric_logger = MetricLogger()
    # model
    model = eval(opt.model)(opt.pretrained,num_classes=opt.num_classes).to(device)
    # model.load_state_dict(torch.load('run/exp12/weights/best.pt'))
    # data
    train_ds = FoodDataset('train',img_size)
    val_ds = FoodDataset('val',img_size)
    test_ds = FoodDataset('test',img_size)
    train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers,)
    val_dl = DataLoader(val_ds, batch_size, False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size, False, num_workers=num_workers)
    # optimizer
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter) and v.bias.requires_grad:  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) and v.weight.requires_grad:  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter) and v.weight.requires_grad:  # weight (with decay)
            g1.append(v.weight)
    if opt.adam:
        optimizer = Adam(g0+g2, lr=lr, betas=(0.9, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0+g2, lr=lr, momentum=0.9)
    optimizer.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
    logger.info(f"{'optimizer:'} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (with decay), {len(g2)} bias")
    # scheduler
    steps = epochs * len(train_dl)
    warmup_steps = opt.warmup_epochs * len(train_dl)
    scheduler = warmup_cosine_schedule(optimizer, warmup_steps, steps)
    # loss func
    if opt.bce_loss:
        loss_func = nn.BCEWithLogitsLoss()
        off_value = opt.label_smooth / opt.num_classes
        on_value = 1. - opt.label_smooth + off_value
    elif opt.center_loss:
        loss_func = CenterLoss()
    else:
        loss_func = LabelSmoothingCrossEntropy(opt.label_smooth)
    # train
    best_acc_top1 = float('-inf')
    scaler = amp.GradScaler()
    for epoch in range(epochs):
        model.train()
        logger.info(f'epoch??? {epoch + 1}/{epochs}')
        logger.info('training:')
        train_loss, val_loss, top1_acc, top5_acc = 0, 0, 0, 0
        for i, (img, label) in tqdm(enumerate(train_dl),total=len(train_dl)):
            img = img.to(device)
            label = label.to(device)
            with amp.autocast():
                img, label1, label2, lam = mixup_data(img, label, opt.mix_up, opt.cut_mix)
                pred = model(img)
                if opt.bce_loss:
                    _label = lam * one_hot(label1, opt.num_classes, on_value, off_value) + (1 - lam) * \
                            one_hot(label2, opt.num_classes, on_value, off_value)
                    loss = loss_func(pred, _label)*opt.num_classes
                else:
                    loss = lam*loss_func(pred, label1)+(1-lam)*loss_func(pred,label2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            train_loss = (train_loss * i + loss.item()) / (i + 1)
        logger.info('validating')
        # val
        with torch.no_grad():
            model.eval()
            for i, (img, label) in tqdm(enumerate(val_dl), total=len(val_dl)):
                img = img.to(device)
                label = label.to(device)
                with amp.autocast():
                    pred = model(img)
                    if opt.bce_loss:
                        _label = one_hot(label, opt.num_classes)
                        loss = loss_func(pred, _label)*opt.num_classes
                    else:
                        loss = loss_func(pred, label)
                _, arg = pred.topk(5, 1)
                label = label.view(-1, 1)
                _top1_acc = (label == arg[:, :1]).sum().item() / label.numel()
                _top5_acc = (label == arg).any(1).sum().item() / label.numel()
                val_loss = (val_loss * i + loss.item()) / (i + 1)
                top1_acc = (top1_acc * i + _top1_acc) / (i + 1)
                top5_acc = (top5_acc * i + _top5_acc) / (i + 1)
        metric_logger.update(lr=scheduler.get_last_lr()[0], train_loss=train_loss, val_loss=val_loss, top1_acc=top1_acc,
                             top5_acc=top5_acc)
        torch.save(model.state_dict(), last)
        logger.info(str(metric_logger))
        if top1_acc > best_acc_top1:
            logger.info('changing best weight...')
            torch.save(model.state_dict(), best)
            best_acc_top1 = top1_acc
    metric_logger.output_csv(txt_res)
    metric_logger.plot(img_res, 3)
    # submit
    model.load_state_dict(torch.load(best))
    arg = np.argmax(metric_logger.meters['top1_acc'])
    top1_acc = metric_logger.meters['top1_acc'][arg]
    top5_acc = metric_logger.meters['top5_acc'][arg]
    logger.info(f'\nsubmit: top1_acc={top1_acc},top5_acc={top5_acc}')
    with torch.no_grad():
        model.eval()
        submit_res = []
        for img in tqdm(test_dl):
            img = img.to(device)
            pred = model(img)
            _, arg = pred.topk(5, 1)
            submit_res.append(arg)
        submit_res = torch.cat(submit_res)
        names = []
        with open(DATA_TXT['test']) as f:
            for line in f.readlines():
                path = line.strip('\n')
                names.append(path)
        sub = pd.DataFrame()
        sub['name'] = names
        sub[['1', '2', '3', '4', '5']] = submit_res.cpu().numpy()
        sub.to_csv(submit,index=False,header=False)


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
