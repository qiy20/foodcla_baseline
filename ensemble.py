import time

import torch
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from timm.models.efficientnet import efficientnet_b2
from timm.models.swin_transformer import swin_base_patch4_window7_224, swin_base_patch4_window7_224_in22k

from baseline import FoodDataset, DATA_TXT


def ensemble(models, weights, dl, device):
    assert len(models) == len(weights)
    res = []
    for model in models:
        _res = []
        with torch.no_grad():
            model.eval()
            for img in tqdm(dl):
                img = img.to(device)
                pred = model(img).softmax(dim=1)
                _res.append(pred)
        res.append(torch.cat(_res))
    res = torch.stack(res)  # n_models x n_images x n_classes
    weights = torch.tensor(weights).view(-1, 1, 1).to(device) # n_models x 1 x 1
    res *= weights
    res = res.mean(0)  # n_images x n_classes
    _, arg = res.topk(5, 1)
    names = []
    with open(DATA_TXT['test']) as f:
        for line in f.readlines():
            path = line.strip('\n')
            names.append(path)
    sub = pd.DataFrame()
    sub['name'] = names
    sub[['1', '2', '3', '4', '5']] = arg.cpu().numpy()
    sub.to_csv(f'run/ensemble_{str(time.time())}.csv', index=False, header=False)


if __name__ == '__main__':
    device = 'cuda:0'
    test_ds = FoodDataset(mode='test')
    test_dl = DataLoader(test_ds, 32, False, num_workers=4)
    model1 = resnext101_32x8d().to(device)
    model1.load_state_dict(torch.load('run/exp11/weights/best.pt', map_location=device))
    model2 = swin_base_patch4_window7_224().to(device)
    model2.load_state_dict(torch.load('run/exp12/weights/best.pt', map_location=device))
    model3=swin_base_patch4_window7_224().to(device)
    model3.load_state_dict(torch.load('run/exp13/weights/best.pt', map_location=device))
    models = [model1, model2,model3]
    ensemble(models, [.5, 1.,1.], test_dl, device)
