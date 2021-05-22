import pytorch_lightning as pl
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
import torchvision.transforms as T
import torch
from RTSD import RTSD
from common import *
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import matplotlib.patches as patches
import tqdm
import neptune.new as neptune
import torch.nn as nn 
from torch.optim.lr_scheduler import StepLR

from prepare import prepare

class RTSD_DataModule(pl.LightningDataModule):
  def __init__(self, batch_size, root, augment=False):
    super().__init__()
    self.batch_size = batch_size
    self.root = root
    prepare(self.root)
    if augment:
      self.train_transform = torchvision.transforms.Compose([
        T.ColorJitter(brightness=.5, hue=.3),
        T.RandomPerspective(distortion_scale=0.2, p=0.25),
        T.RandomEqualize(p=0.25),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
      self.train_dataset = RTSD(train=True, root_path=f'{self.root}data/', transform=self.train_transform)
    else:
      self.train_dataset = RTSD(train=True, root_path=f'{self.root}data/')

    self.val_dataset = RTSD(train=False, root_path=f'{self.root}data/')

  def collate(batch):
    return tuple(zip(*batch))

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       collate_fn=RTSD_DataModule.collate,
                                       num_workers=4,
                                       shuffle=True,
                                       pin_memory=True)

  def test_dataloader(self):
    return torch.utils.data.DataLoader(self.val_dataset,
                                       batch_size=self.batch_size,
                                       collate_fn=RTSD_DataModule.collate,
                                       num_workers=4,
                                       shuffle=True,
                                       pin_memory=True)
  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.val_dataset,
                                       batch_size=self.batch_size,
                                       collate_fn=RTSD_DataModule.collate,
                                       num_workers=4,
                                       shuffle=True,
                                       pin_memory=True)

class LitModel50(pl.LightningModule):
  def __init__(self, optimizer, learning_rate, optimizer_params, root):
    super().__init__()
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.optimizer_params = optimizer_params
    self.root = root
    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 198
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    self.precision = 0
    self.recall = 0
    self.batch_cnt = 0
    self.cur_epoch = 0

  def forward(self, x):
    return self.model(x)
    
  def training_step(self, batch, batch_idx):
    images, targets = batch
    loss_dict = self.model(images, targets)
    print(loss_dict)
    losses = sum(loss for loss in loss_dict.values())
    self.logger.experiment.log_metric('train/loss', losses)
    return losses
  
  def on_training_epoch_end(self):

    with open(f'{self.root}/models/model50.pt', 'wb') as f:
      torch.save(self.model, f)
    
    print("Saved model!")

    self.cur_epoch += 1
    self.logger.experiment.log_artifact(f'{self.root}/models/model50.pt')

  def configure_optimizers(self):
    return self.optimizer(self.parameters(), self.learning_rate,
                          **self.optimizer_params)
