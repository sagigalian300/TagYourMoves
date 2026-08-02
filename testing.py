import cv2
import os
from ultralytics import YOLO
import sys

# ==========================================
# 1. Setup and Loading
# ==========================================
YOLO_MODEL_PATH = './model/yolo_best.pt'
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"[ERROR] Could not find the model weights at: {YOLO_MODEL_PATH}")
    sys.exit(1)

model = YOLO(YOLO_MODEL_PATH)
print("[SUCCESS] Model loaded successfully")

VIDEO_PATH = './assets/videos/3.mp4'
if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Could not find the video at: {VIDEO_PATH}")
    sys.exit(1)
print("[SUCCESS] Video found")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
    sys.exit(1)

# Colors for each class (in OpenCV BGR format)
# 0: ball (Green), 1: player (Blue), 2: rim (Red)
COLORS = {
    0: (0, 255, 0),    # Ball - Green
    1: (255, 0, 0),    # Player - Blue
    2: (0, 0, 255)     # Rim - Red
}

CLASS_NAMES = {0: 'ball', 1: 'player', 2: 'rim'}

print("\n[INFO] Starting live visualization...")
print("[INFO] Controls:")
print("  - Press 'q' to exit")
print("  - Press 'SPACE' to pause / unpause")

is_paused = False

while cap.isOpened():
    if not is_paused:
        success, frame = cap.read()
        if not success:
            print("\n[INFO] End of video reached.")
            break
        
        # Run YOLO tracking on the current frame
        # results = model.predict(source=frame, conf=0.51, imgsz=1280, verbose=False)
        results = model.track(source=frame, conf=0.51, imgsz=1280, persist=True, verbose=False)
        
        # Draw predictions on the frame
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = float(box.conf[0].item())
            
            x1, y1, x2, y2 = map(int, coords)
            color = COLORS.get(cls_id, (255, 255, 255))
            class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            
            # Extract tracking ID if it exists and format it for the label
            track_id_text = ""
            if box.id is not None and cls_id == 1:  # Only append ID for players
                track_id = int(box.id[0].item())
                track_id_text = f" ID:{track_id}"
            
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw the text (object class + Tracking ID + confidence score)
            label = f"{class_name}{track_id_text} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # If it's a ball (cls_id == 0), draw a prominent yellow dot in its center for visualization
            if cls_id == 0:
                center_x = int((coords[0] + coords[2]) / 2.0)
                center_y = int((coords[1] + coords[3]) / 2.0)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)

        # Display the current frame number on the screen
        cv2.putText(frame, f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame using OpenCV
    cv2.imshow("YOLO Live Visualization", frame)
    
    # Handle keypresses (1ms delay between frames)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Pressing 'q' will exit the loop
        print("\n[INFO] User interrupted visualization.")
        break
    elif key == ord(' '):  # Spacebar will toggle pause/unpause
        is_paused = not is_paused

cap.release()
cv2.destroyAllWindows()
print("[SUCCESS] Done!")