from pathlib import Path
import argparse
import yaml
import logging

import torch
from torch import nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, resnet50, resnet101
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils import init_seeds, increment_path, MetricLogger

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

DATA_TXT = {'train': 'train_qtcom.txt', 'val': 'val_qtcom.txt', 'test': 'test_qtcom.txt'}
DATA_DIR = {'train': 'Train_qtc', 'val': 'val', 'test': 'test_new'}


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--save_dir', default='run/exp')
    opt = parser.parse_intermixed_args()
    return opt


class FoodDataset(Dataset):
    def __init__(self, mode, augment=True, infer_size=224):  # mode 'train','val' or 'test'
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
        if augment:
            self.trans = transforms.Compose([transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1),
                                                                     scale=(.9, 1.1), shear=(-10, 10)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, ),
                                             ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = cv2.imread(self.img_paths[item])
        img = cv2.resize(img, self.infer_size)
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
    save_dir, device, epochs, lr, weight_decay, batch_size, num_workers = Path(
        opt.save_dir), opt.device, opt.epochs, opt.lr, opt.weight_decay, opt.batch_size, opt.num_workers
    save_dir = increment_path(save_dir)
    w = save_dir / 'weights'  # weights dir, /==.joinpath()
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    txt_res, img_res = save_dir / 'res.txt', save_dir / 'res.png'
    submit = save_dir / 'submit.csv'
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    metric_logger = MetricLogger()
    model = eval(opt.model)(num_classes=1000).to(device)
    train_ds = FoodDataset(mode='train')
    val_ds = FoodDataset(mode='val')
    test_ds = FoodDataset(mode='test')
    train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size, False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size, False, num_workers=num_workers)
    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)
    if opt.adam:
        optimizer = Adam(g0, lr=lr, betas=(0.9, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=lr, momentum=0.9)
    optimizer.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    logger.info(f"{'optimizer:'} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (with decay), {len(g2)} bias")
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss_func = nn.CrossEntropyLoss()
    min_val_loss = float('inf')
    for epoch in range(epochs):
        logger.info(f'epoch： {epoch + 1}/{epochs}')
        logger.info('training:')
        train_loss, val_loss, top1_acc, top5_acc = 0, 0, 0, 0
        for i, (img, label) in tqdm(enumerate(train_dl),total=len(train_dl)):
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            loss = loss_func(pred, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss = (train_loss * i + loss.item()) / (i + 1)
        logger.info('validating')
        with torch.no_grad():
            model.eval()
            for i, (img, label) in tqdm(enumerate(val_dl), total=len(val_dl)):
                img = img.to(device)
                label = label.to(device)
                pred = model(img)
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
        scheduler.step()
        torch.save(model.state_dict(), last)
        logger.info(str(metric_logger))
        if val_loss < min_val_loss:
            logger.info('changing best weight...')
            torch.save(model.state_dict(), best)
            min_val_loss = val_loss
    metric_logger.output_csv(txt_res)
    metric_logger.plot(img_res, 3)
    # submit
    model.load_state_dict(torch.load(best))
    arg = np.argmin(metric_logger.meters['val_loss'])
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