import cv2
import os
from ultralytics import YOLO
import sys

from src.objects.FrameSnapshot import FrameSnapshot
from src.OfflineRepair import OfflineRepair

from src.StateMachines.ShotStateMachine import ShotStateMachine
from src.StateMachines.PassStateMachine import PassStateMachine

# names: {0: 'ball', 1: 'player', 2: 'rim'}
YOLO_MODEL_PATH = './model/yolo_best.pt'
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"[ERROR] Could not find the model weights at: {YOLO_MODEL_PATH}")
else:
    model = YOLO(YOLO_MODEL_PATH)
    print("[SUCCESS] Model loaded successfully")


VIDEO_PATH = './assets/videos/3.mp4'
if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Could not find the video at: {VIDEO_PATH}")
    sys.exit(1)
print("[SUCCESS] video found")


cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


state_machines = [ShotStateMachine(fps), PassStateMachine(fps)]
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
    #results = model.predict(source=frame, conf=0.51, imgsz=1280, verbose=False)
    results = model.track(source=frame, conf=0.51, imgsz=1280, persist=True, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]

        if cls_id == 0:   # Ball
            current_snapshot.set_ball((coords[0] + coords[2]) / 2.0, (coords[1] + coords[3]) / 2.0, coords[0], coords[1], coords[2], coords[3])
        elif cls_id == 1: # Player
            player_id = None
            if box.id is not None:
                player_id = int(box.id[0].item())
            current_snapshot.add_player(coords[0], coords[1], coords[2], coords[3], player_id)
        elif cls_id == 2: # Rim
            current_snapshot.add_rim(coords[0], coords[1], coords[2], coords[3])

    # Check if the ball is missing in the current frame
    if current_snapshot.is_ball_missing():
        buffer.append(current_snapshot)
        continue

    # We get here only if the ball is detected in the current frame
    new_ball_position = (current_snapshot.ball.x_center, current_snapshot.ball.y_center)

    if len(buffer) > 0 and last_ball_position is not None:
        repaired_coords = offline_repairer.get_repaired_coords(last_ball_position, new_ball_position, len(buffer))
                
        for buffered_snap, rep_coord in zip(buffer, repaired_coords):
            buffered_snap.set_ball(rep_coord[0], rep_coord[1], 0, 0, 0, 0)

            for state_machine in state_machines:
                state_machine.update(buffered_snap)

    buffer = []
    for state_machine in state_machines:
        state_machine.update(current_snapshot)

    last_ball_position = new_ball_position

cap.release()
cv2.destroyAllWindows()

print(f"\n[SUCCESS] Done! Total processed frames: {frame_count}")
