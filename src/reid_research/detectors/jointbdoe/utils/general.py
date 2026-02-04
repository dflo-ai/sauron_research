# Minimal general.py - only functions needed for JointBDOE inference

import math
import logging
from pathlib import Path
import torch
import torchvision
import numpy as np


def set_logging(rank=-1):
    """Set up logging."""
    logging.basicConfig(format="%(message)s", level=logging.INFO if rank in [-1, 0] else logging.WARN)


def check_file(file):
    """Search for file if not found."""
    if Path(file).is_file() or file == '':
        return file
    else:
        raise FileNotFoundError(f'File not found: {file}')


def check_img_size(img_size, s=32):
    """Verify img_size is a multiple of stride s."""
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print(f'WARNING: img_size {img_size} must be multiple of stride {s}, updating to {new_size}')
    return new_size


def make_divisible(x, divisor):
    """Returns x evenly divisible by divisor."""
    return math.ceil(x / divisor) * divisor


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                        multi_label=False, labels=(), max_det=300, num_angles=1):
    """
    Non-Maximum Suppression (NMS) on inference results with body orientation support.

    Returns:
        list of detections, on (n,6+num_angles) tensor per image [xyxy, conf, cls, angles...]
    """
    nc = prediction.shape[2] - 5 - num_angles  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 4096  # max box width and height
    max_nms = 30000  # max boxes for NMS

    output = [torch.zeros((0, 6 + num_angles), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence filter

        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:5+nc] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Get angles (body orientation)
        angles = x[:, 5+nc:5+nc+num_angles]  # shape: (n, num_angles)

        # Detections matrix nx(6+num_angles) (xyxy, conf, cls, angles...)
        if multi_label:
            i, j = (x[:, 5:5+nc] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), angles[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:5+nc].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), angles), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]

    return output


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)."""
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
