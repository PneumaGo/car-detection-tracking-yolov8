import cv2
from collections import defaultdict
from .model_loader import load_model
from .config import *
from .utils import get_color_for_id

model = load_model()
object_centers_history = defaultdict(lambda: [])
counted_object_ids = set()
class_counters = {}

def process_video_and_count(video_path, progress=None):
    global object_centers_history, counted_object_ids, class_counters

    # Reset globals
    object_centers_history = defaultdict(lambda: [])
    counted_object_ids.clear()
    class_counters.clear()
    for cls_id in model.names.keys():
        class_counters[cls_id] = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file."

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    counting_line_y = int(frame_height * 2 / 3)
    output_video_path = TEMP_OUTPUT_VIDEO
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if progress:
            progress(frame_count / total_frames, desc="Processing video...")

        results = model.track(frame,
                              iou=0.5,
                              conf=0.4,
                              imgsz=480,
                              verbose=False,
                              persist=True,
                              tracker="botsort.yaml",
                              half=True)

        annotated_frame = results[0].plot()
        cv2.line(annotated_frame, (COUNTING_LINE_START_X, counting_line_y),
                 (COUNTING_LINE_END_X, counting_line_y), (0, 255, 0), 2)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
            current_frame_ids_with_cls = set()

            for box, obj_id, cls_idx in zip(boxes, ids, class_indices):
                current_frame_ids_with_cls.add((obj_id, cls_idx))
                center_x = (box[0] + box[2]) // 2
                center_y = box[3]
                object_centers_history[obj_id].append((center_x, center_y))
                if len(object_centers_history[obj_id]) > MAX_HISTORY_LENGTH:
                    object_centers_history[obj_id].pop(0)

                if len(object_centers_history[obj_id]) >= 2:
                    prev_y = object_centers_history[obj_id][-2][1]
                    curr_y = object_centers_history[obj_id][-1][1]
                    if prev_y <= counting_line_y and curr_y > counting_line_y and (obj_id, cls_idx) not in counted_object_ids:
                        if cls_idx in class_counters:
                            class_counters[cls_idx] += 1
                            counted_object_ids.add((obj_id, cls_idx))

            all_current_ids_only = {pair[0] for pair in current_frame_ids_with_cls}
            ids_to_remove = [obj_id for obj_id in object_centers_history if obj_id not in all_current_ids_only]
            for obj_id in ids_to_remove:
                del object_centers_history[obj_id]

        # Display counters
        y_offset = 100
        for cls_idx in sorted(class_counters.keys()):
            class_name = model.names.get(cls_idx, f"Class {cls_idx}")
            count = class_counters[cls_idx]
            cv2.putText(annotated_frame, f"{class_name}: {count}", (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            y_offset += 30

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Summary
    counts_summary = "Final Counting Results:\n"
    total_counted = 0
    for cls_idx in sorted(class_counters.keys()):
        class_name = model.names.get(cls_idx, f"Class {cls_idx}")
        count = class_counters[cls_idx]
        counts_summary += f"{class_name}: {count}\n"
        total_counted += count
    counts_summary += f"\nTotal Counted: {total_counted}"

    return output_video_path, counts_summary
