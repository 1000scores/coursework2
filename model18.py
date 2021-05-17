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

      self.train_dataset = RTSD(train=True, root_path=f"{self.root}data/", transform=self.train_transform)
      
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

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.val_dataset,
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


class LitModel18(pl.LightningModule):
  def __init__(self, optimizer, learning_rate, optimizer_params, root):
    super().__init__()
    self.optimizer = optimizer
    self.learning_rate = learning_rate
    self.optimizer_params = optimizer_params
    self.root = root
    resnet18 = torchvision.models.resnet18(pretrained=True)
    backbone = nn.Sequential(*(list(resnet18.children())[:-2])) 
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                  aspect_ratios=((0.5, 1.0, 2.0),))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

    self.model = FasterRCNN(backbone,
                   num_classes=198,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

    self.precision = 0
    self.recall = 0
    self.batch_cnt = 0
    self.cur_epoch = 0



  def forward(self, x):
    return self.model(x)

    
  def training_step(self, batch, batch_idx):
    images, targets = batch
    loss_dict = self.model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    self.logger.experiment.log_metric('train/loss', losses)
    return losses
  
  def on_train_epoch_end(self, outputs):
    self.scheduler.step()
    torch.save({
    'epoch': self.cur_epoch,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    }, f'{self.root}/models/model18_state_dict_epoch{self.cur_epoch}.pt')

    with open(f'{self.root}/models/model18_epoch{self.cur_epoch}.pt', 'wb') as f:
      torch.save(self.model, f)
    
    print("Saved model!")
    self.logger.experiment.log_artifact(f'{self.root}/models/model18_epoch{self.cur_epoch}.pt')
    self.cur_epoch += 1
    

  def validation_step(self, batch, batch_idx):
    images, targets = batch
    bboxes = self.model(images)
    intersected_bboxes = intersect(bboxes)
    self.recall += get_recall(intersected_bboxes, targets)
    self.precision += get_precision(intersected_bboxes, targets)
    self.batch_cnt += 1
  
  def on_validation_epoch_end(self):
    self.logger.experiment.log_metric('test/precision',
                                      self.precision / self.batch_cnt)
    
    self.logger.experiment.log_metric('test/recall',
                                      self.recall / self.batch_cnt)

    f1 = (self.precision + self.recall) / (2 * self.precision * self.recall)
    
    self.logger.experiment.log_metric('test/f1',
                                      f1)
    self.precision = 0
    self.recall = 0

  def configure_optimizers(self):

    self.optimizer = self.optimizer(self.parameters(), self.learning_rate,
                          **self.optimizer_params)

    self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5)

    return self.optimizer
