from datetime import timedelta
from pathlib import Path
import torch
import numpy as np


def get_iou(gt_mask, pred_mask, ignore_label=-1):

    pred_mask = (pred_mask > 0.5).int()

    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    iou = intersection / union if union != 0 else 1.0
    return iou


def get_dice(gt_mask, pred_mask, ignore_label=-1):
    # Apply a threshold to the predicted mask
    pred_mask = (pred_mask > 0.5).int()

    # Convert torch tensors to numpy arrays if necessary
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()

    # Generate masks to ignore the ignored labels and identify object of interest
    valid_mask = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    # Calculate intersection and union for the Dice coefficient
    intersection = np.logical_and(pred_mask, obj_gt_mask).sum()
    union = pred_mask.sum() + obj_gt_mask.sum()

    # Calculate Dice coefficient
    dice = (2. * intersection) / union if union != 0 else 1.0

    return dice


def get_dice_array(gt_mask, pred_mask, ignore_label=-1):
    # Apply a threshold to the predicted mask
    pred_mask = (pred_mask > 0.5).astype(int)

    # Generate masks to ignore the ignored labels and identify object of interest
    valid_mask = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    # Calculate intersection and union for the Dice coefficient
    intersection = np.logical_and(pred_mask, obj_gt_mask).sum()
    union = pred_mask.sum() + obj_gt_mask.sum()

    # Calculate Dice coefficient
    dice = (2. * intersection) / union if union != 0 else 1.0

    return dice