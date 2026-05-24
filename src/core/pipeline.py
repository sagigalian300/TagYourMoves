import cv2
import torch
import numpy as np
from ultralytics import YOLO

from src.core.trajectory_repair import TrajectoryRepairer
from src.features.extraction import KinematicFeatureExtractor
from src.utils.video_manager import VideoManager
from src.utils.visualizer import Visualizer

class BasketballInferencePipeline:
    def __init__(self, yolo_model_path, lstm_model_path, scaler_path, confidence_threshold=0.85):
        """
        Main Orchestrator Pipeline to tie CV models, sequential logic, and video streams together.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models and structural services
        self.yolo_model = YOLO(yolo_model_path)
        
        # PyTorch 2.6+ compliance: explicitly set weights_only=False to load full custom models
        self.lstm_model = torch.load(lstm_model_path, map_location=self.device, weights_only=False)
        self.lstm_model.eval()
        
        self.repairer = TrajectoryRepairer(window_size=90)
        self.extractor = KinematicFeatureExtractor(scaler_path)
        self.visualizer = Visualizer()
        
        self.conf_threshold = confidence_threshold
        self.frame_buffer = [] 

    def run(self, input_video_path, output_video_path):
        """
        Executes end-to-end sequential prediction over the target video stream.
        """
        video = VideoManager(input_video_path, output_video_path)
        width, height = video.get_dimensions()
        frame_counter = 0

        print(f"🎬 Processing stream: {input_video_path} via Abstract Live-Infrastructure...")

        while True:
            ret, frame = video.read_frame()
            if not ret:
                break

            frame_counter += 1
            
            results = self.yolo_model.predict(source=frame, conf=0.25, imgsz=1280, verbose=False)
            boxes = results[0].boxes

            current_ball_center = None
            current_rim_center = None
            ball_box_coords = None
            rim_box_coords = None

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    if cls == 0:    
                        current_ball_center = np.array([cx, cy])
                        ball_box_coords = [x1, y1, x2, y2]
                    elif cls == 2:  
                        current_rim_center = np.array([cx, cy])
                        rim_box_coords = [x1, y1, x2, y2]

            if current_rim_center is None:
                current_rim_center = np.array([width / 2.0, height / 2.0])

            self.repairer.add_frame(current_ball_center, current_rim_center)
            
            self.frame_buffer.append({
                "image": frame,
                "ball_box": ball_box_coords,
                "rim_box": rim_box_coords
            })

            if len(self.repairer.raw_buffer) >= self.repairer.window_size:
                if self.repairer.raw_buffer[90 - 1]["ball"] is not None or frame_counter % 30 == 0:
                    
                    # Create a deep local slice copy of the current 90 frames to safely handle edge cases
                    window_slice = [{"ball": f["ball"], "rim": f["rim"]} for f in self.repairer.raw_buffer[:90]]
                    known_indices = [idx for idx, f in enumerate(window_slice) if f["ball"] is not None]

                    if len(known_indices) == 0:
                        # Fallback case: If the ball was never detected at all, fill completely with rim baseline
                        for f in window_slice:
                            f["ball"] = f["rim"]
                    else:
                        # 1. Interpolate internal occlusion gaps between known points
                        for i in range(len(known_indices) - 1):
                            idx_s, idx_e = known_indices[i], known_indices[i+1]
                            if idx_e - idx_s > 1:
                                p_s = window_slice[idx_s]["ball"]
                                p_e = window_slice[idx_e]["ball"]
                                steps = idx_e - idx_s
                                for s in range(1, steps):
                                    alpha = s / steps
                                    window_slice[idx_s + s]["ball"] = (1 - alpha) * p_s + alpha * p_e

                        # 2. Backfill leading Nones (if the video starts with occlusion)
                        first_idx = known_indices[0]
                        if first_idx > 0:
                            p_first = window_slice[first_idx]["ball"]
                            for idx in range(first_idx):
                                window_slice[idx]["ball"] = p_first

                        # 3. Forward-fill trailing Nones (if the slice ends with occlusion)
                        last_idx = known_indices[-1]
                        if last_idx < 89:
                            p_last = window_slice[last_idx]["ball"]
                            for idx in range(last_idx + 1, 90):
                                window_slice[idx]["ball"] = p_last

                    # Extract mathematical kinematic features from the bulletproof window
                    lstm_input = self.extractor.extract_and_scale(window_slice)

                    window_tensor = torch.FloatTensor(lstm_input).to(self.device)
                    with torch.no_grad():
                        shot_prob = self.lstm_model(window_tensor).item()
                        is_shot = shot_prob >= self.conf_threshold

                    active_data = self.frame_buffer.pop(0)
                    self.repairer.raw_buffer.pop(0) 
                    
                    out_frame = active_data["image"]
                    
                    active_ball_box = active_data["ball_box"]
                    if active_ball_box is None and window_slice[0]["ball"] is not None:
                        bcx, bcy = window_slice[0]["ball"]
                        active_ball_box = [bcx - 15, bcy - 15, bcx + 15, bcy + 15] 

                    self.visualizer.draw_predictions(out_frame, active_ball_box, active_data["rim_box"])
                    if is_shot:
                        self.visualizer.draw_shot_alert(out_frame, shot_prob, width, height)

                    video.write_frame(out_frame)

            if frame_counter % 100 == 0:
                print(f"⏳ Stabilized and rendered {frame_counter} frames...")

        while len(self.frame_buffer) > 0:
            active_data = self.frame_buffer.pop(0)
            video.write_frame(active_data["image"])

        video.release()
        print(f"\n🎉 Pipeline Complete! Visual output secured at: {output_video_path}")
