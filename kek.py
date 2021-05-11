import os
from os import listdir
from os.path import isfile, join
import tqdm
import pandas as pd
import json

all = None
with open('data/labels.json', 'r') as f:
    all = json.load(f)

cnt = 0
for key in all.keys():
    print(all[key])
    cnt += 1
    if cnt == 100:
        break