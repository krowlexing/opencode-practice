from ultralytics import YOLO
import cv2
import numpy as np

from group import detect_groups_task
from helmet import detect_helmets_task
from cv_utils import write_text, draw_boxes, draw_box, extract_boxed_images, save_image

def find_person_id(model):
    names = model.names
    for k, v in names.items():
        if v == "person":
            return k
    return None

def filter_person(boxes):
    return filter(lambda box: int(box.cls[0]) == 0, boxes)

def detect_people_task(image, box_list):
    n = len(box_list)
    image_with_boxes = draw_boxes(image, box_list)
    output = write_text(image_with_boxes, str(n))
    save_image("output_person_detection.jpg", output)

model = YOLO("yolov8x.pt")

result = model("input.jpeg", conf=0.2, classes=[0])[0]

filtered_boxes = list(filter_person(result.boxes))

detect_people_task(result.orig_img.copy(), filtered_boxes)
detect_groups_task(result.orig_img.copy(), filtered_boxes)
detect_helmets_task(result.orig_img.copy(), filtered_boxes)
