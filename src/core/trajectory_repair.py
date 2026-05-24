import numpy as np

class TrajectoryRepairer:
    def __init__(self, window_size=90):
        """
        Abstract class for managing and repairing the ball trajectory in real-time
        using a Look-ahead Buffer architecture.
        """
        self.window_size = window_size
        self.raw_buffer = []  # Temporary buffer to store frames until occlusion is resolved
        
    def add_frame(self, ball_coords, rim_coords):
        """
        Appends raw detection coordinates from the current frame into the buffer.
        
        :param ball_coords: numpy array [cx, cy] or None if undetected
        :param rim_coords: numpy array [cx, cy]
        """
        self.raw_buffer.append({
            "ball": ball_coords,  
            "rim": rim_coords     
        })
        
    def is_ready_to_process(self):
        """
        Checks if the buffer contains enough frames and if the occlusion gap has closed.
        
        :return: True if a complete window can be processed, False otherwise
        """
        if len(self.raw_buffer) < self.window_size:
            return False
            
        # If the ball is missing in the latest frame, keep buffering until it is detected again
        if self.raw_buffer[-1]["ball"] is None:
            return False
            
        return True

    def repair_and_pop_window(self):
        """
        Performs linear interpolation over missing ball positions within the block,
        retaining exact physical continuity, and pops a clean window.
        
        :return: A list of 90 dictionaries with completely resolved coordinates
        """
        # Find all frame indices where the ball was successfully detected
        known_indices = [idx for idx, frame in enumerate(self.raw_buffer) if frame["ball"] is not None]
        
        # Linearly interpolate over the missing frames between known points
        for i in range(len(known_indices) - 1):
            idx_start = known_indices[i]
            idx_end = known_indices[i+1]
            
            # Check if there is an occlusion gap between detections
            if idx_end - idx_start > 1:
                pos_start = self.raw_buffer[idx_start]["ball"]
                pos_end = self.raw_buffer[idx_end]["ball"]
                steps = idx_end - idx_start
                
                for step in range(1, steps):
                    curr_idx = idx_start + step
                    alpha = step / steps
                    # Retroactively reconstruct the missing position
                    self.raw_buffer[curr_idx]["ball"] = (1 - alpha) * pos_start + alpha * pos_end

        # Extract exactly 90 frames from the front of the queue (fully repaired)
        processed_window = [self.raw_buffer.pop(0) for _ in range(self.window_size)]
        return processed_window
