import os
from PIL import Image
import torch
from common import *
import tqdm
import json
import torchvision
from PIL import Image, ImageDraw

def test_curmodel(image , targets, path = f'best_models/resnet50_SGD.pt', threshold_approve=0.3, threshold_intersect=0.1, device="cpu"):
    model = torch.load(path, map_location=device)
    model.eval()
    recall = 0
    precision = 0
    batches = 0

    bboxes_pred = model(images)
    bboxes_pred = intersect(bboxes_pred, threshold_approve, threshold_intersect)
    '''recall += get_recall(bboxes_pred, targets)
    precision += get_precision(bboxes_pred, targets)
    print("Final:")
    print(f'Recall = {recall}')
    print(f'Precision = {precision}')'''

    return bboxes_pred[0]

def read_img(path):
    transform = torchvision.transforms.ToTensor()
    return transform(Image.open(path).convert('RGB'))

paths = ['autosave01_02_2012_09_14_02.jpg',
         'autosave01_02_2012_09_31_04.jpg',
         'autosave01_02_2012_10_28_25.jpg',
         'autosave01_02_2012_10_40_31.jpg',
         'autosave01_02_2012_11_29_37.jpg',
         'autosave09_10_2012_09_27_24_2.jpg',
         'autosave09_10_2012_13_33_01_0.jpg'
         ]
# 
# autosave01_02_2012_09_28_29.jpg
# autosave01_02_2012_09_31_04.jpg
# autosave01_02_2012_10_28_25.jpg
# autosave01_02_2012_10_40_31.jpg
# autosave01_02_2012_11_29_37.jpg

it = 0
for path in paths:
    labels = None
    with open(f'data/labels.json', 'r') as f:
        labels = json.load(f)

    target = {
        'boxes': list(),
        'labels': list()
    }
    if path in labels:
        for label in labels[path]:
            target['boxes'].append([label[0], label[1], label[2], label[3]])
            target['labels'].append(label[4])
        target['boxes'] = torch.FloatTensor(target['boxes'])
        target['labels'] = torch.LongTensor(target['labels'])

    image = read_img(f'test_images/{path}')
    images = []
    images.append(image)
    targets = []
    targets.append(target)
    bboxes = test_curmodel(image, targets)
    image = Image.open(f'test_images/{path}')
    draw = ImageDraw.Draw(image)
    print('True:')
    print('Labels: ', target['labels'])
    print('Predict:')
    print('Labels: ', bboxes['labels'])
    print('Scores: ', bboxes['scores'])

    for box in bboxes['boxes']:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=2)
    image.save(f'results/{it}.jpg')
    it += 1
