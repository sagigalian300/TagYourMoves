
import cv2
import numpy as np

class OnlineKalmanFilter:
    def __init__(self, max_missing_frames=7):
        # 4 state variables (x, y, vx, vy), 2 measurement variables (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Measurement matrix: maps state space to measurement space
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        
        # Transition matrix: constant velocity dynamics model
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        
        # Process noise covariance (Q): High velocity noise allows instant adapting to bounces
        self.kf.processNoiseCov = np.array([[1e-3, 0,    0,   0],
                                            [0,    1e-3, 0,   0],
                                            [0,    0,    0.5, 0],
                                            [0,    0,    0,   0.5]], np.float32)
        
        # Measurement noise covariance (R): Low noise forces filter to instantly lock onto YOLO shifts
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        self.initialized = False
        self.missing_count = 0
        self.max_missing_frames = max_missing_frames

    def process(self, ball_coords):
        """
        Processes a single frame in real-time.
        Returns estimated (x, y) coordinates and status string.
        """
        if ball_coords is not None:
            self.missing_count = 0  # Reset missing counter upon valid detection
            measured = np.array([[np.float32(ball_coords[0])], [np.float32(ball_coords[1])]], np.float32)
            
            if not self.initialized:
                self.kf.statePost = np.array([[measured[0][0]], [measured[1][0]], [0], [0]], np.float32)
                self.initialized = True
            
            self.kf.predict()
            estimate = self.kf.correct(measured)
            return (int(estimate[0][0]), int(estimate[1][0])), "Detected"
        else:
            if self.initialized:
                self.missing_count += 1
                
                # Guardrail: Stop predicting if occlusion lasts too long to prevent drift
                if self.missing_count > self.max_missing_frames:
                    return None, "Missing"
                
                prediction = self.kf.predict()
                self.kf.statePost = prediction
                return (int(prediction[0][0]), int(prediction[1][0])), "Kalman Prediction"
            
            return None, "Missing"
