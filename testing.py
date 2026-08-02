import cv2
import os
from ultralytics import YOLO
import sys

# ==========================================
# 1. הגדרות וטעינה
# ==========================================
YOLO_MODEL_PATH = './model/yolo_best.pt'
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"[ERROR] Could not find the model weights at: {YOLO_MODEL_PATH}")
    sys.exit(1)

model = YOLO(YOLO_MODEL_PATH)
print("[SUCCESS] Model loaded successfully")

VIDEO_PATH = './assets/videos/5.mp4'
if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Could not find the video at: {VIDEO_PATH}")
    sys.exit(1)
print("[SUCCESS] Video found")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Could not open video file: {VIDEO_PATH}")
    sys.exit(1)

# צבעים לכל מחלקה (בפורמט BGR של OpenCV)
# 0: ball (ירוק), 1: player (כחול), 2: rim (אדום)
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
        
        # הרצת YOLO על הפריים הנוכחי
        results = model.predict(source=frame, conf=0.51, imgsz=1280, verbose=False)
        
        # ציור התחזיות על הפריים
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            conf = float(box.conf[0].item())
            
            x1, y1, x2, y2 = map(int, coords)
            color = COLORS.get(cls_id, (255, 255, 255))
            class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            
            # ציור הריבוע (Bounding Box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ציור הטקסט (שם האובייקט + רמת הביטחון)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # אם זה הכדור (cls_id == 0), נצייר גם נקודה בולטת במרכז שלו להמחשה
            if cls_id == 0:
                center_x = int((coords[0] + coords[2]) / 2.0)
                center_y = int((coords[1] + coords[3]) / 2.0)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1) # נקודה צהובה במרכז

        # הצגת מספר הפריים הנוכחי על המסך
        cv2.putText(frame, f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # הצגת הפריים בעזרת OpenCV
    cv2.imshow("YOLO Live Visualization", frame)
    
    # טיפול במקשים (השהייה של 1 מילישנייה בין פריימים)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # לחיצה על q תסגור את הריצה
        print("\n[INFO] User interrupted visualization.")
        break
    elif key == ord(' '):  # רווח יעשה פאוז / ריווח
        is_paused = not is_paused

cap.release()
cv2.destroyAllWindows()
print("[SUCCESS] Done!")