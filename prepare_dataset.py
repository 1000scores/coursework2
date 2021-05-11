from os import listdir
from os.path import isfile, join
import pandas as pd
import tqdm
from PIL import Image
import torchvision
import torch
import os

files = listdir('data/rtsd') 
labels = pd.read_csv('data/full-gt.csv')
cnt = 1
for file in tqdm.tqdm(files):
    path = f'data/rtsd/{file}'
    img = Image.open(path).convert('RGB')
    width, height = img.size
    if width != 1280 or height != 720:
        os.remove(path)
        cnt += 1
print('kek')
print(cnt)