import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
import argparse
from os import listdir
from common import intersect
import json

def read_img(path):
    transform = torchvision.transforms.ToTensor()
    return transform(Image.open(path).convert('RGB'))

def run_model(image, device='cuda'):
    model = torch.load('best_models/model50_epochs3.pt', map_location=torch.device(device))
    model.eval()
    transform = torchvision.transforms.ToTensor()
    output = model([transform(image.convert('RGB')).to(device)])
    output = intersect(output, threshold_approve=0.7, threshold_intersect=0.3)
    output = output[0]
    draw = ImageDraw.Draw(image)
    
    to_sign_class = None
    with open('data/index_to_sign_class.json', 'r') as f:
        to_sign_class = json.load(f)

    labels = []
    for l in output['labels']:
        labels.append(to_sign_class[str(l)])

    for box in output['boxes']:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=2)
    
    print('Scores:')
    print(output['scores'])    
    print('Success!')
    return (image, labels, output['boxes'])

