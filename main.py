import argparse

import numpy as np
import pandas as pd
import os

import ast
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import os
import time
import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2
from torch.optim import Adam


def get_label(csv_file):
    return label_dict[csv_file.split('/')[-1].replace(' ', '_')[:-4]]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class DoodleDataset(Dataset):
    def __init__(self, csv_file, root_dir, mode='train', nrows=1000, skiprows=None,
                 size=256, thickness=2, transform=None):
        super(DoodleDataset, self).__init__()

        self.thickness = thickness
        self.root_dir = root_dir
        self.mode = mode
        self.size = size
        self.transform = transform
        file = os.path.join(self.root_dir, csv_file)
        self.data = pd.read_csv(file, usecols=['drawing'], nrows=nrows, skiprows=skiprows)
        if self.mode == 'train':
            self.label = get_label(csv_file)

    @staticmethod
    def draw(raw_strokes, size=256, thickness=2, color_by_time=True):
        img = np.zeros((255, 255), np.uint8)
        for t, stroke in enumerate(raw_strokes):
            for i in range(1, len(stroke[0])):
                color = 255 - min(t, 10) * 13 if color_by_time else 255
                cv.line(img, (stroke[0][i - 1], stroke[1][i - 1]), (stroke[0][i], stroke[1][i]), color, thickness)

        if size != 255:
            img = cv.resize(img, (size, size))

        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_strokes = ast.literal_eval(self.data.drawing[idx])
        sample = self.draw(raw_strokes, self.size, self.thickness)

        if self.transform:
            sample = self.transform(sample)
        if self.mode == 'train':
            # sample[None] equivalent to numpy.newaxis
            return (sample[None] / 255).astype('float32'), self.label
        else:
            return (sample[None] / 255).astype('float32')

def mapk(output, target, k=3):
    """
    Computes the mean average precision at k.

    Parameters
    ----------
    output (torch.Tensor): A Tensor of predicted elements.
                           Shape: (N,C)  where C = number of classes, N = batch size
    target (torch.int): A Tensor of elements that are to be predicted.
                        Shape: (N) where each value is  0≤targets[i]≤C−1
    k (int, optional): The maximum number of predicted elements

    Returns
    -------
    score (torch.float):  The mean average precision at k over the output
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).float()
        for i in range(k):
            correct[i].mul_(1.0 / (i + 1))
        score = correct.view(-1).sum(0)
        score.mul_(1.0 / batch_size)
        return score

if __name__ == '__main__':
    label_dict = {}
    path = r'dataset\train_simplified'
    filenames = sorted(glob.glob(os.path.join(path, '*.csv')))
    print(filenames)
    for i, fn in enumerate(filenames):
        # print(fn[:-4].split('\\'))
        # print(fn[:-4].split('//', '/')[-1].replace(' ', '_'))
        label_dict[fn[:-4].split('\\')[-1].replace(' ', '_')] = i
    dec_dict = {v: k for k, v in label_dict.items()}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 128
    epochs = 1
    input_nc = 64
    lr = 2e-3
    image_size = 224
    select_nrows = 10000
    num_classes = 340
    print_freq = 1000

    filename_pth = 'pretrained/checkpoint_mobilenetv2.pth'

    model = mobilenet_v2()


    def squeeze_weights(m):
        m.weight.data = m.weight.data.sum(1, keepdim=True)
        m.in_channels = 1


    model.features[0][0].apply(squeeze_weights)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes),
    )
    print(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5000, 12000, 18000], gamma=0.5)
    model.load_state_dict(torch.load(filename_pth, map_location=device))
    model.eval()



    testset = DoodleDataset('test_simplifide_five.csv', 'dataset/test_simplified', mode='test', nrows=None, size=image_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    labels = np.empty((0, 3))
    for img in tqdm.tqdm(testloader):
        img = img.to(device)
        output = model(img)
        _, pred = output.topk(3, 1, True, True)
        labels = np.concatenate([labels, pred.cpu()], axis=0)

    dec_dict = {v: k for k, v in label_dict.items()}
    # print(dec_dict)

    file = 'dataset/test_simplified/test_simplifide_five.csv'
    submission = pd.read_csv(file)


    submission.drop(['countrycode', 'drawing'], axis=1, inplace=True)
    submission['word'] = ''
    for i, label in enumerate(labels):
        submission.word.iloc[i] = " ".join([dec_dict[l] for l in label])
    submission.to_csv('preds_mobilenetv2.csv')