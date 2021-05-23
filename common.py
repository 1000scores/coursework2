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
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torch import nn
from RTSD import RTSD
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
import matplotlib.pyplot as plt
from sklearn.metrics import auc

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

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def intersect(targets, threshold_approve=0, threshold_intersect=0.1):
    for ind in range(len(targets)): 
        bboxes = targets[ind]['boxes']
        labels = targets[ind]['labels']
        scores = targets[ind]['scores']
        
        boxes_res = list()
        labels_res = list()
        scores_res = list()
        for ind0 in range(len(bboxes)):
            winner = True
            for ind1 in range(len(bboxes)):
                box0 = bboxes[ind0].cpu().detach().numpy()
                box1 = bboxes[ind1].cpu().detach().numpy()
                label0 = labels[ind0]
                label1 = labels[ind1]
                score0 = scores[ind0]
                score1 = scores[ind1]
                if (score0 < threshold_approve) or (score0 < score1 and bb_intersection_over_union(box0, box1) > threshold_intersect):
                    winner = False
                    break
            if winner:
                boxes_res.append(bboxes[ind0].cpu().detach().numpy())
                labels_res.append(labels[ind0].item())
                scores_res.append(scores[ind0].item())
        targets[ind]['boxes'] = boxes_res
        targets[ind]['labels'] = labels_res
        targets[ind]['scores'] = scores_res
        
    return targets

def get_true_false_negative(batch_pred, batch_true, threshold_intersect):

    assert len(batch_pred) == len(batch_true)

    fn = 0
    tn = 0
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
                if label_pred == label_true and bb_intersection_over_union(box_pred, box_true) > threshold_intersect:
                    found = True
                    break
            if found:
                tn += 1
            else:
                fn += 1
    return (tn, fn)


def get_true_false_positive(batch_pred, batch_true, threshold_intersect):

    assert len(batch_pred) == len(batch_true)

    tp = 0
    fp = 0
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
                if label_pred == label_true and bb_intersection_over_union(box_pred, box_true) > threshold_intersect:
                    found = True
                    break
            if found:
                tp += 1
            else:
                fp += 1

    return (tp, fp)

def get_distance(bboxes, max_dist=30, w_image = 1280, h_height=720):
  ans = list()
  for bbox in bboxes:
    w_sign = bbox[2] - bbox[0]
    h_sign = bbox[3] - bbox[1]
    ans.append(max_dist * (1 - (w_sign/w_image)))
  return ans

def cnt_bboxes(test_loader):
    device = 'cpu'
    cnt_bboxes = 0
    for index, (images, targets) in tqdm.tqdm(enumerate(test_loader)):
        images = [img.to(device) for img in images]
        targets = [trgt for trgt in targets]
        for trgt in targets:
            trgt['boxes'] = trgt['boxes'].to(device)
            trgt['labels'] = trgt['labels'].to(device)
            cnt_bboxes += len(trgt['boxes'])
    print(f'Num of bboxes = {cnt_bboxes}')

def test_model(path, test_loader, threshold_approve, threshold_intersect, device="cuda"):
    model = torch.load(path, map_location=device)
    model.eval()
    batches = 0
    print("batches: ", len(test_loader))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index, (images, targets) in enumerate(test_loader):
        images = [img.to(device) for img in images]
        targets = [trgt for trgt in targets]
        for trgt in targets:
            trgt['boxes'] = trgt['boxes'].to(device)
            trgt['labels'] = trgt['labels'].to(device)
        bboxes_pred = model(images)
        bboxes_pred = intersect(bboxes_pred, threshold_approve, threshold_intersect)
        tp_fp = get_true_false_positive(bboxes_pred, targets, threshold_intersect)
        tp += tp_fp[0]
        fp += tp_fp[1]
        tn_fn = get_true_false_negative(bboxes_pred, targets, threshold_intersect)
        tn += tn_fn[0]
        fn += tn_fn[1]
        batches += 1
    
    print(f'tp = {tp} ')
    print(f'fp = {fp} ')
    print(f'fn = {fn} ')

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (precision + recall) / (2 * precision * recall)
    #print(f'Precision: {precision}')
    #print(f'Recall: {recall}')
    #print(f'F1: {f1}')

    return (tp, fp, tn, fn)

def roc_auc(path, test_loader, ROOT, device='cuda'):
    threshs = np.linspace(0, 1, 15, endpoint=False)
    X = []
    y = []
    for t in threshs:
        tp, fp, tn, fn = test_model(path, test_loader, t, 0.3, device=device)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        X.append(fpr)
        y.append(tpr)
    X = [0] + X + [1]
    y = [0] + y + [1] 
    plt.grid(True)
    plt.plot(X, y)
  
    return (auc(X, y), X, y)


