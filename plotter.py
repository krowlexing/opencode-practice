# This is taken from ultralytics/engine/result.py
# Modified to omit detected class
# Removed unused flags and code

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER, SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode

def plot(
    self,
    conf=True,
    line_width=None,
    font_size=None,
    font="Arial.ttf",
    pil=False,
    img=None,
    im_gpu=None,
    labels=True,
    boxes=True,
    masks=True,
    probs=True,
    show=False,
):
    """
    Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

    Args:
        conf (bool): Whether to plot the detection confidence score.
        line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
        font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
        font (str): The font to use for the text.
        pil (bool): Whether to return the image as a PIL Image.
        img (numpy.ndarray): Plot to another image. if not, plot to original image.
        im_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
        kpt_radius (int, optional): Radius of the drawn keypoints. Default is 5.
        kpt_line (bool): Whether to draw lines connecting keypoints.
        labels (bool): Whether to plot the label of bounding boxes.
        boxes (bool): Whether to plot the bounding boxes.
        masks (bool): Whether to plot the masks.
        probs (bool): Whether to plot classification probability.
        show (bool): Whether to display the annotated image directly.
        save (bool): Whether to save the annotated image to `filename`.
        filename (str): Filename to save image to if save is True.

    Returns:
        (numpy.ndarray): A numpy array of the annotated image.

    Example:
        ```python
        from PIL import Image
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt')
        results = model('bus.jpg')  # results list
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.show()  # show image
            im.save('results.jpg')  # save image
        ```
    """
    if img is None and isinstance(self.orig_img, torch.Tensor):
        img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

    names = self.names
    is_obb = self.obb is not None
    pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
    pred_masks, show_masks = self.masks, masks
    pred_probs, show_probs = self.probs, probs
    annotator = Annotator(
        deepcopy(self.orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names,
    )

    # Plot Detect results
    if pred_boxes is not None and show_boxes:
        for d in reversed(pred_boxes):
            id = None if d.id is None else int(d.id.item())
            name = ("" if id is None else f"id:{id} ")
            label = (f"{name}" if conf else name) if labels else None
            box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
            annotator.box_label(box, label, color=colors(0, True), rotated=is_obb)

    return annotator.result()