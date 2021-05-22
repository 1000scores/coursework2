import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

root = ''
df = pd.read_csv(f'{root}data/full-gt.csv')

all = None
with open(f'{root}data/labels.json', 'r') as f:
    all = json.load(f)

sign_class_to_index = None
with open(f'{root}data/sign_class_to_index.json', 'r') as f:
    sign_class_to_index = json.load(f)


stat = df['sign_class']
stat.hist(bins=198, xlabelsize=0, grid=False)
plt.show()

