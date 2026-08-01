import cv2
import os
from ultralytics import YOLO
import sys

from src.FrameSnapshot import FrameSnapshot
from src.OfflineRepair import OfflineRepair

# names: {0: 'ball', 1: 'player', 2: 'rim'}
YOLO_MODEL_PATH = './model/yolo_best.pt'
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"[ERROR] Could not find the model weights at: {YOLO_MODEL_PATH}")
else:
    model = YOLO(YOLO_MODEL_PATH)
    print("[SUCCESS] Model loaded successfully")


VIDEO_PATH = './assets/videos/1.mp4'
if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Could not find the video at: {VIDEO_PATH}")
    sys.exit(1)
print("[SUCCESS] video found")


cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

offline_repairer = OfflineRepair()
last_ball_position = None
buffer = []
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    # New frame snapshot
    current_snapshot = FrameSnapshot(frame_index=frame_count)

    # Filling the snapshot with detected objects
    results = model.predict(source=frame, conf=0.51, imgsz=1280, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]

        if cls_id == 0:   # Ball
            current_snapshot.ball_center = ((coords[0] + coords[2]) / 2.0, (coords[1] + coords[3]) / 2.0)
        elif cls_id == 1: # Player
            current_snapshot.players_boxes.append(coords)
        elif cls_id == 2: # Rim
            current_snapshot.rims_boxes.append(coords)

    # Check if the ball is missing in the current frame
    if current_snapshot.is_ball_missing:
        buffer.append(current_snapshot)
        continue

    # We get here only if the ball is detected in the current frame
    new_ball_position = current_snapshot.ball_center

    if len(buffer) > 0 and last_ball_position is not None:
        repaired_coords = offline_repairer.get_repaired_coords(last_ball_position, new_ball_position, len(buffer))
                
        for buffered_snap, rep_coord in zip(buffer, repaired_coords):
            buffered_snap.ball_center = rep_coord
            print(f"[REPAIRED] Frame {buffered_snap.frame_index}: Ball interpolated at (X={rep_coord[0]:.1f}, Y={rep_coord[1]:.1f})")
            # here update the state machine for each buffered snapshot
            
    buffer = []
    print(f"[DETECTED] Frame {current_snapshot.frame_index}: Ball found by YOLO at (X={new_ball_position[0]:.1f}, Y={new_ball_position[1]:.1f})")
    last_ball_position = new_ball_position

cap.release()
cv2.destroyAllWindows()

print(f"\n[SUCCESS] Done! Total processed frames: {frame_count}")
