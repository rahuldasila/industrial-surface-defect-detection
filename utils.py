import torch
import numpy as np

def dice_score(preds, targets, smooth=1e-6):

    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    dice = (2.0 * intersection + smooth) / ( preds.sum() + targets.sum() + smooth)

    return dice.item()

def iou_score(preds, targets, smooth=1e-6):

    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()