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

def run_model(image):
    device = 'cpu'
    model = torch.load('best_models/resnet50_SGD.pt', map_location=torch.device('cpu'))
    model.eval()
    transform = torchvision.transforms.ToTensor()
    output = model([transform(image.convert('RGB'))])
    output = intersect(output, threshold_approve=0.3, threshold_intersect=0.3)
    output = output[0]
    draw = ImageDraw.Draw(image)
    labels = output['labels']
    for box in output['boxes']:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=2)
    print('Scores:')
    print(output['scores'])    
    print('Success!')
    return (image, labels)

    

if __name__ == '__main__':
    true_labels = None
    with open(f'data/labels.json', 'r') as f:
        true_labels = json.load(f)

    with open('data/test_with_spaces.txt', 'r') as f:
        files = f.read().split("\n")
        for ind in range(len(files)):
            with Image.open(f'data/rtsd_test/{files[ind]}') as img:
                image, labels = run_model(img)
                print("Predicted:")
                print(labels)
                print("True:")
                print(true_labels[files[ind]])
                image.show()
            break

