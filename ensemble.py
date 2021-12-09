from baseline import *


def ensemble(models, weights, dls, device):
    assert len(models) == len(weights) == len(dls)
    res = []
    for model,dl in zip(models,dls):
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
    res = res.mean(0)  # n_images x n
    # _classes
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
    dl1 = DataLoader(FoodDataset(mode='test'), 16, False, num_workers=4)
    model1 = swin_base_patch4_window7_224_in22k(num_classes=1000).to(device)
    model1.load_state_dict(torch.load('run/exp19/weights/best.pt', map_location=device))
    dl2 = DataLoader(FoodDataset(mode='test', infer_size=384), 16, False, num_workers=4)
    model2 = swin_base_patch4_window12_384_in22k(num_classes=1000).to(device)
    model2.load_state_dict(torch.load('run/exp22/weights/best.pt', map_location=device))
    model3 = swin_large_patch4_window7_224_in22k(num_classes=1000).to(device)
    model3.load_state_dict(torch.load('run/exp21/weights/best.pt', map_location=device))
    models = [model1, model2, model3]
    dls = [dl1, dl2, dl1]
    ensemble(models, [.5, 1., .5], dls, device)
