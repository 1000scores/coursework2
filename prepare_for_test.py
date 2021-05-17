import pandas as pd
import json
import tqdm
from os import listdir
import os
from sklearn.model_selection import train_test_split

def prepare_test():

    all = None
    with open(f'data/labels.json', 'r') as f:
        all = json.load(f)
    print(len(all))

    files = listdir(f'data/rtsd_test')
    print(len(files))

    X = files
    X_train, X_test = train_test_split(X, test_size=0.001)

    print(len(X_test))


    with open(f'data/test_with_spaces.txt', 'w') as f:
        for ind in range(len(X_test)):
            if ind + 1 < len(X_test):
                f.write(f'{X_test[ind]}\n')
            else:
                f.write(f'{X_test[ind]}')




if __name__ == '__main__':
    prepare_test()
