"""JointBDOE postprocessing utilities ported from jointbdoe/utils/general.py.

Provides Non-Maximum Suppression and coordinate scaling for YOLO-style detection.
"""
import time

import numpy as np
import torch
import torchvision


def xywh2xyxy(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format.

    Args:
        x: Boxes in xywh format (N, 4)

    Returns:
        Boxes in xyxy format (N, 4)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(
    boxes: torch.Tensor | np.ndarray, shape: tuple[int, int]
) -> None:
    """Clip bounding boxes to image shape (height, width).

    Args:
        boxes: Bounding boxes in xyxy format (N, 4+)
        shape: Image shape as (height, width)

    Note: Modifies boxes in-place.
    """
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(
    img1_shape: tuple[int, int],
    coords: torch.Tensor | np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple | None = None,
) -> torch.Tensor | np.ndarray:
    """Rescale coords (xyxy) from img1_shape to img0_shape.

    Args:
        img1_shape: Shape of processed image (H, W)
        coords: Coordinates to scale (N, 4+) with xyxy in first 4 columns
        img0_shape: Shape of original image (H, W)
        ratio_pad: Optional (ratio, pad) from letterbox

    Returns:
        Scaled coordinates
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    nl = coords.shape[0]
    if nl == 0:
        return coords

    coords = coords.reshape((nl, -1, 2))
    coords[..., 0] -= pad[0]
    coords[..., 1] -= pad[1]
    coords /= gain
    coords = coords.reshape(nl, -1)
    clip_coords(coords, img0_shape)

    return coords


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU of box1 with all boxes in box2.

    Args:
        box1: (N, 4) tensor of boxes
        box2: (M, 4) tensor of boxes

    Returns:
        (N, M) IoU matrix
    """

    def box_area(box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    area1 = box_area(box1)
    area2 = box_area(box2)

    inter = (
        torch.min(box1[:, None, 2:], box2[:, 2:])
        - torch.max(box1[:, None, :2], box2[:, :2])
    ).clamp(0).prod(2)

    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: list | None = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: tuple = (),
    max_det: int = 300,
    num_angles: int = 3,
) -> list[torch.Tensor]:
    """Run Non-Maximum Suppression (NMS) on inference results.

    Args:
        prediction: Model output tensor (B, N, 5 + num_classes + num_angles)
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        classes: Filter by class indices
        agnostic: Class-agnostic NMS
        multi_label: Multiple labels per box
        labels: Prior labels for autolabelling
        max_det: Maximum detections per image
        num_angles: Number of orientation angle outputs

    Returns:
        List of detections per image, each tensor (N, 6 + num_angles):
        [x1, y1, x2, y2, conf, cls, angles...]
    """
    nc = prediction.shape[2] - 5 - num_angles  # Number of classes
    xc = prediction[..., 4] > conf_thres  # Candidates

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid confidence threshold {conf_thres}"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}"

    # Settings
    max_wh = 4096  # Maximum box width/height
    max_nms = 30000  # Maximum boxes into NMS
    time_limit = 10.0  # Seconds to quit after
    redundant = True  # Require redundant detections
    multi_label &= nc > 1  # Multiple labels per box
    merge = False  # Use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + num_angles), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # Image index, image inference
        x = x[xc[xi]]  # Apply confidence filter

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        # Compute conf = obj_conf * cls_conf
        x[:, 5:-num_angles] *= x[:, 4:5]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls, angles)
        if multi_label:
            i, j = (x[:, 5:-num_angles] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat(
                (box[i], x[i, j + 5, None], j[:, None].float(), x[i, -num_angles:]), 1
            )
        else:  # Best class only
            conf, j = x[:, 5:-num_angles].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:, -num_angles:]), 1)[
                conf.view(-1) > conf_thres
            ]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # Classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # Boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)

        if i.shape[0] > max_det:
            i = i[:max_det]

        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break

    return output
