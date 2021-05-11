import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from RTSD import RTSD
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import matplotlib.patches as patches
import tqdm


def plot_images_bb(batch_pred, batch_true):
    images = batch_pred[0]
    targets_pred = batch_pred[1]
    targets_true = batch_true
    for ind, img in enumerate(images): 
        img = img.to("cpu")
        img = rearrange(img, 'c h w -> h w c')
        target_pred = targets_pred[ind]['boxes']
        target_true = targets_true[ind]['boxes']
        fig, ax = plt.subplots()
        ax.imshow(img)
        for box in target_pred:
            x0 = box[0]
            y0 = box[1]
            w = box[2] - x0
            h = box[3] - y0
            rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        '''for box in target_true:
            x0 = box[0]
            y0 = box[1]
            w = box[2] - x0
            h = box[3] - y0
            rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='g', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)'''
        plt.show()


# [x0, y0, x1, y1]
def IoU(box0: list, box1: list):
    x0, y0 = max(box0[0], box1[0]), max(box0[1], box1[1])
    x1, y1 = min(box0[2], box1[2]), min(box0[3], box1[3])
    s0 = abs((box0[0] - box0[2]) * (box0[1] - box0[3]))
    s1 = abs((box1[0] - box1[2]) * (box1[1] - box1[3]))
    s_intersect = abs((x1 - x0) * (y1 - y0))
    return s_intersect / (s0 + s1 - s_intersect)


def intersect(batch, threshold_approve = 0.5, threshold_intersect = 0.3):
    images = batch[0]
    targets = batch[1]
    for ind in range(len(targets)): 
        bboxes = targets[ind]['boxes']
        labels = targets[ind]['labels']
        scores = targets[ind]['scores']
        
        boxes_res = list()
        labels_res = list()
        scores_res = list()

        for ind0 in range(len(bboxes)):
            winner = True
            for ind1 in range(ind0 + 1, len(bboxes)):
                box0 = bboxes[ind0].cpu().detach().numpy()
                box1 = bboxes[ind1].cpu().detach().numpy()
                label0 = labels[ind0]
                label1 = labels[ind1]
                score0 = scores[ind0]
                score1 = scores[ind1]
                if (score0 < threshold_approve) or (label0 == label1 and score0 < score1 and IoU(box0, box1) > threshold_intersect):
                    winner = False
                    break
            if winner:
                boxes_res.append(bboxes[ind0].cpu().detach().numpy())
                labels_res.append(labels[ind0].item())
                scores_res.append(scores[ind0].item())
        targets[ind]['boxes'] = boxes_res
        targets[ind]['labels'] = labels_res
        targets[ind]['scores'] = scores_res
        
    return [images, targets]


def get_recall(batch_pred, batch_true, threshold = 0.5):

    assert len(batch_pred) == len(batch_true)

    cnt_recall = 0.0
    for ind in range(len(batch_pred)):
        targets_pred, targets_true = batch_pred[ind], batch_true[ind]
        bboxes_pred, bboxes_true = targets_pred['boxes'], targets_true['boxes']
        labels_pred, labels_true = targets_pred['labels'], targets_true['labels']
        cnt_found = 0.0
        for ind_true, box_true in enumerate(bboxes_true):
            found = False
            for ind_pred, box_pred in enumerate(bboxes_pred):
                label_pred = labels_pred[ind_pred]
                label_true = labels_true[ind_true]
                if label_pred == label_true and IoU(box_pred, box_true) > threshold:
                    found = True
                    break
            if found:
                cnt_found += 1.0
        
        cnt_recall += cnt_found / len(bboxes_true)

    return cnt_recall / len(batch_true)


def get_precision(batch_pred, batch_true, threshold = 0.5):

    assert len(batch_pred) == len(batch_true)

    cnt_precision = 0.0
    for ind in range(len(batch_pred)):
        targets_pred, targets_true = batch_pred[ind], batch_true[ind]
        bboxes_pred, bboxes_true = targets_pred['boxes'], targets_true['boxes']
        labels_pred, labels_true = targets_pred['labels'], targets_true['labels']
        cnt_found = 0.0
        for ind_pred, box_pred in enumerate(bboxes_pred):
            found = False
            for ind_true, box_true in enumerate(bboxes_true):
                label_pred = labels_pred[ind_pred]
                label_true = labels_true[ind_true]
                if label_pred == label_true and IoU(box_pred, box_true) > threshold:
                    found = True
                    break
            if found:
                cnt_found += 1
        
        cnt_precision += cnt_found / len(bboxes_true)

    return cnt_precision / len(batch_true)