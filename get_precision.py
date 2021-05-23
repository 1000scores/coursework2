from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torch import nn
from RTSD import RTSD
from common import *
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import matplotlib.patches as patches
import tqdm
import neptune.new as neptune
from pytorch_lightning.loggers.neptune import NeptuneLogger
from model50 import *
from model18 import *

ROOT = ''
val_dataset = RTSD(train=False, root_path=f'{ROOT}data/', with_spaces=True)
batch_size = 4
def collate(batch):
    return tuple(zip(*batch))

def test_dataloader():
    return torch.utils.data.DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    collate_fn=collate,
                                    shuffle=True)


tp, fp, tn, fn = test_model(f'{ROOT}best_models/resnet50_SGD.pt',
           test_dataloader(),
           threshold_approve = 0.6,
           threshold_intersect = 0.2,
           device='cpu')

