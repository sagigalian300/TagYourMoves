import cv2
import os
from ultralytics import YOLO
import sys
from src.filters.kalman_repair import OnlineKalmanFilter
from src.filters.offline_repair import OfflineRepair
from src import config

YOLO_MODEL_PATH = './model/yolo_best.pt'

if not os.path.exists(YOLO_MODEL_PATH):
    print(f"[ERROR] Could not find the model weights at: {YOLO_MODEL_PATH}")
else:
    model = YOLO(YOLO_MODEL_PATH)
    print("[SUCCESS] Model loaded successfully")


VIDEO_PATH = './assets/videos/2.mp4'
OUTPUT_PATH = './assets/detected_examples/output_processed.mp4'

classes_to_plot = []
if config.SHOW_BALL_BOX:    classes_to_plot.append(0)
if config.SHOW_PLAYER_BOX:  classes_to_plot.append(1) 
if config.SHOW_BASKET_BOX:  classes_to_plot.append(2)  
    
if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Could not find the video at: {VIDEO_PATH}")
    sys.exit(1)

print("[SUCCESS] video found")

if config.USE_KALMAN:
    kalman = OnlineKalmanFilter()
    
if config.USE_OFFLINE_REPAIR:
    offline_repairer = OfflineRepair()
    frames_to_repair = []
    last_valid_coords = None
    new_valid_coords = None
    
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = None
if config.SAVE_OUTPUT:
    print(f"[INFO] Save mode activated, Processing and saving to: {OUTPUT_PATH}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
else:
    print("[INFO] Live mode activated, Opening visual stream window")
    print("[INFO] Press 'q' on your keyboard inside the video window to stop")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    results = model.predict(source=frame, conf=0.51, imgsz=1280, verbose=False)

    plot_mask = [int(box.cls[0].item()) in classes_to_plot for box in results[0].boxes]
    res_plotted = results[0][plot_mask].plot()
    
    raw_ball_coords = None
    for box in results[0].boxes:
        if int(box.cls[0].item()) == 0: 
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            raw_ball_coords = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            break
                
    # ---------------- Using KALMANM ----------------
    if config.USE_KALMAN:        
        repaired_coords, status = kalman.process(raw_ball_coords)
        
        # Draw visual indicator if YOLO failed but Kalman provided a trajectory prediction
        if raw_ball_coords is None and repaired_coords is not None:
            cx, cy = repaired_coords
            cv2.circle(res_plotted, (cx, cy), 12, (0, 140, 255), 5)
            cv2.putText(res_plotted, "Kalman", (cx - 25, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    # ------------------------------------------------
    
    # ---------------- Using offline repair ----------------
    if config.USE_OFFLINE_REPAIR:
        if raw_ball_coords is None:
            frames_to_repair.append(res_plotted)     # adding the frame for future repair
            continue                                 # dont draw the frame, will be drone when repaired
        else: 
            if len(frames_to_repair) == 0:            # no frames to repair
                last_valid_coords = raw_ball_coords
            else:                                    # found the new_valid_coords -> repair
                new_valid_coords = raw_ball_coords

                if last_valid_coords is not None:    # edge case - what if the ball is missing from the first frame ?! there are frames to repair but not last_valid_coords
                    repaired_coords_list = offline_repairer.get_repaired_coords_from_last_to_new(last_valid_coords, new_valid_coords, len(frames_to_repair))                
                    offline_repairer.show_repaired_frames(frames_to_repair, repaired_coords_list, save_output=config.SAVE_OUTPUT, video_writer=out, fps=fps)
                else:
                    offline_repairer.show_repaired_frames(frames_to_repair, [None]*len(frames_to_repair), save_output=config.SAVE_OUTPUT, video_writer=out, fps=fps)                    
                
                frames_to_repair = []
                last_valid_coords = new_valid_coords
                new_valid_coords = None
    # --------------------------------------------------------
    
    if config.SAVE_OUTPUT:
        out.write(res_plotted)
        if frame_count % 100 == 0:
            print(f"[INFO] Processed and saved {frame_count} frames")
    else:
        cv2.imshow("TagYourMoves - Live Local Inference", res_plotted)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Execution stopped by user.")
            break
            
# Clear the buffer & display frames couldnt repair (because ball not found until the end) in case using OFFLINE_REPAIR
if config.USE_OFFLINE_REPAIR and len(frames_to_repair) > 0:
    print(f"[INFO] Flushing remaining {len(frames_to_repair)} frames at the end of video")
    dummy_coords = [last_valid_coords if last_valid_coords is not None else (0,0)] * len(frames_to_repair)
    offline_repairer.show_repaired_frames(frames_to_repair, dummy_coords, save_output=config.SAVE_OUTPUT, video_writer=out, fps=fps)
    
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

print(f"\n[SUCCESS] Done! Total processed frames: {frame_count}")
