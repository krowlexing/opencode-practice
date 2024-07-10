from collections import defaultdict

import cv2
import numpy as np

from types import MethodType
from ultralytics import YOLO
from plotter import plot
from helmet import detect_helmets
from cv_utils import draw_box

model = YOLO("yolov8x.pt")

video_path = "video-tracking.mp4"
input_file = cv2.VideoCapture(video_path)
input_fps = input_file.get(cv2.CAP_PROP_FPS)
input_fourcc = input_file.get(cv2.CAP_PROP_FOURCC)
input_frame_width = int(input_file.get(cv2.CAP_PROP_FRAME_WIDTH))
input_frame_height = int(input_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_frame_size = (input_frame_width, input_frame_height)

out_all = cv2.VideoWriter("output.mp4", cv2.VideoWriter.fourcc(*'mp4v'), input_fps, input_frame_size)
out_no_helmet = cv2.VideoWriter("output_no_helmet.mp4", cv2.VideoWriter.fourcc(*'mp4v'), input_fps, input_frame_size)

track_history = defaultdict(lambda: [])

def draw_tracking_no_helmet(out_image, boxes, track_ids):
    for box, track_id in zip(boxes, track_ids):
        draw_box(out_image, box, (0, 0, 255))
        x, y, w, h = box.xywh[0]
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 200:
            track.pop(0)

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out_image, [points], isClosed=False, color=(255, 0, 0), thickness=10)

n_tracks = 0

while input_file.isOpened():
    success, frame = input_file.read()

    if success:
        results = model.track(frame, persist=True, conf=0.2, classes=[0])
        results[0].plot = MethodType(plot, results[0])
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids_cpu = results[0].boxes.id.int().cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        n_tracks = max(n_tracks, *track_ids)

        annotated_all_frame = results[0].plot(conf=False, probs=False, line_width=2)        
        out_all.write(annotated_all_frame)

        no_helmet_frame = frame.copy()
        
        helmets = detect_helmets(frame, results[0].boxes)
        no_helmets = list(map(lambda x: not x, helmets))
        
        results[0].boxes = results[0].boxes[no_helmets]
        no_helmet_track_ids = track_ids_cpu[no_helmets].tolist()

        draw_tracking_no_helmet(no_helmet_frame, results[0].boxes, no_helmet_track_ids)
        out_no_helmet.write(no_helmet_frame)

    else:
        break

input_file.release()
out_all.release()
out_no_helmet.release()

print("found ", n_tracks, " people")