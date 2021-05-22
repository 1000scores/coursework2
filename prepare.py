import pandas as pd
import json
import tqdm
from os import listdir
import os
from sklearn.model_selection import train_test_split

def prepare(root):
  df = pd.read_csv(f'{root}data/full-gt.csv')
  print(df.head())
  all = dict()

  sign_class_to_index = dict()
  index_to_sign_class = dict()
  ind = 0
  for row in df.iterrows():
      cls = row[1]['sign_class']
      if cls not in sign_class_to_index:
          sign_class_to_index[cls] = ind
          index_to_sign_class[ind] = cls
          ind += 1


  for row in df.iterrows():
      if row[1]['filename'] in all:
          all[row[1]['filename']].append((row[1]['x_from'], row[1]['y_from'], row[1]['x_from'] + row[1]['width'],
                                          row[1]['y_from'] + row[1]['height'], sign_class_to_index[row[1]['sign_class']]))
      else:
          all[row[1]['filename']] = [(row[1]['x_from'], row[1]['y_from'], row[1]['x_from'] + row[1]['width'],
                                      row[1]['y_from'] + row[1]['height'], sign_class_to_index[row[1]['sign_class']])]

  with open(f'{root}data/labels.json', 'w') as f:
      json.dump(all, f)

  with open(f'{root}data/sign_class_to_index.json', 'w') as f:
      json.dump(sign_class_to_index, f)

  with open(f'{root}data/index_to_sign_class.json', 'w') as f:
      json.dump(index_to_sign_class, f)

  all = None
  with open(f'{root}data/labels.json', 'r') as f:
      all = json.load(f)
  print(len(all))

  files = listdir(f'{root}data/rtsd')
  print(len(files))
  cnt = 0
  for file in files:
      if file not in all:
          os.remove(f'{root}data/rtsd/{file}')
          cnt += 1
          
  print(f'cnt = {cnt}')

  X = list(all.keys())
  X_train, X_test = train_test_split(X, test_size=0.1)

  print(len(X_train))
  print(len(X_test))
  with open(f'{root}data/train.txt', 'w') as f:
      for ind in range(len(X_train)):
          if ind + 1 < len(X_train):
              f.write(f'{X_train[ind]}\n')
          else:
              f.write(f'{X_train[ind]}')

  with open(f'{root}data/test.txt', 'w') as f:
      for ind in range(len(X_test)):
          if ind + 1 < len(X_test):
              f.write(f'{X_test[ind]}\n')
          else:
              f.write(f'{X_test[ind]}')

if __name__ == '__main__':
  ROOT = ''
  prepare(ROOT)
