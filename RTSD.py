import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
import pandas as pd
from PIL import Image
import torchvision
import json

class RTSD(Dataset):

    def __init__(self, train: bool, root_path = 'data', transform = torchvision.transforms.ToTensor()):
        self.transform = transform
        self.root_path = root_path
        if train:
            with open(f'{root_path}/train.txt', 'r') as f:
                self.img_paths = f.read().split('\n')
        else:
            with open(f'{root_path}/test.txt', 'r') as f:
                self.img_paths = f.read().split('\n')

        self.labels = None
        with open(f'{root_path}/labels.json', 'r') as f:
            self.labels = json.load(f)


    def __getitem__(self, index):
        path = self.img_paths[index]
        target = dict()
        target['boxes'] = list()
        target['labels'] = list()
        for label in self.labels[path]:
            target['boxes'].append([label[0], label[1], label[2], label[3]])
            target['labels'].append(label[4])
        
        target['boxes'] = torch.FloatTensor(target['boxes'])
        target['labels'] = torch.LongTensor(target['labels'])

        return self.read_img(f'{self.root_path}/rtsd/{path}'), target

    def __len__(self):
        return len(self.img_paths)

    def read_img(self, path):
        return self.transform(Image.open(path).convert('RGB'))

        
