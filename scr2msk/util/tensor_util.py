import torch.nn.functional as F
import numpy as np

def compute_tensor_iu(seg, gt):
    intersection = (seg & gt).float().sum()
    union = (seg | gt).float().sum()

    return intersection, union

def compute_np_iu(seg, gt):
    intersection = (seg & gt).astype(np.float32).sum()
    union = (seg | gt).astype(np.float32).sum()

    return intersection, union

def compute_tensor_iou(seg, gt):
    intersection, union = compute_tensor_iu(seg, gt)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou 

# STM
def pad_divide_by(in_img, d, in_size=None):
    if in_size is None:
        h, w = in_img.shape[-2:]
    else:
        h, w = in_size

    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out = F.pad(in_img, pad_array)
    return out, pad_array