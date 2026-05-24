import numpy as np
import joblib

class KinematicFeatureExtractor:
    def __init__(self, scaler_path):
        """
        Abstract class for calculating and normalizing basketball kinematic features
        from a continuous frame window.
        """
        self.scaler = joblib.load(scaler_path)
        self.feature_dim = 7

    def extract_and_scale(self, cleaned_window):
        """
        Transforms a cleaned frame window into a normalized matrix ready for LSTM inference.
        
        :param cleaned_window: List of 90 dictionaries containing 'ball' and 'rim' coordinates
        :return: Standardized numpy array with shape (1, 90, 7) containing batch dimension
        """
        seq_len = len(cleaned_window)
        features = np.zeros((seq_len, self.feature_dim))

        # Step A: Extract fixed rim anchor from the very last frame for baseline stability
        rim_anchor = cleaned_window[-1]["rim"]

        # Step B: Distance features (dx, dy, dist)
        for i, frame in enumerate(cleaned_window):
            bx, by = frame["ball"]
            features[i, 0] = bx - rim_anchor[0]  # dx
            features[i, 1] = by - rim_anchor[1]  # dy
            features[i, 2] = np.sqrt(features[i, 0]**2 + features[i, 1]**2)  # dist

        # Step C: Velocity derivatives (vx, vy)
        for i in range(1, seq_len):
            features[i, 3] = features[i, 0] - features[i-1, 0]  # vx
            features[i, 4] = features[i, 1] - features[i-1, 1]  # vy

        # Step D: Acceleration derivatives (ax, ay)
        for i in range(2, seq_len):
            features[i, 5] = features[i, 3] - features[i-1, 3]  # ax
            features[i, 6] = features[i, 4] - features[i-1, 4]  # ay

        # Step E: Apply production standardization parameters (Fitted StandardScaler)
        features_reshaped = features.reshape(-1, self.feature_dim)
        features_scaled = self.scaler.transform(features_reshaped)
        
        # Step F: Reshape to add batch dimension -> (1, 90, 7)
        features_final = features_scaled.reshape(1, seq_len, self.feature_dim)
        
        return features_final
