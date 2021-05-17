import os
from os import listdir
from os.path import isfile, join
import tqdm
import pandas as pd
import json
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

ROOT = ''
device = 'cpu'

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
num_classes = 198
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

checkpoint = torch.load(f'best_models_state_dict/resnet50_SGD.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

with open('best_models/resnet50_SGD.pt', 'wb') as f:
    torch.save(model, f)