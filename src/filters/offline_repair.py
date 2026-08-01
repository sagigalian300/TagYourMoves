import cv2
import numpy as np

class OfflineRepair:
    def __init__(self):
        """
        Initializes the offline trajectory repair engine using linear interpolation.
        """
        pass

    def get_repaired_coords_from_last_to_new(self, last_coords, new_coords, num_frames):
        """
        Calculates a linear interpolation path between the last valid coordinates 
        and the newly detected coordinates across the missing frames gap.
        """
        if num_frames <= 0:
            return []
        x_points = np.linspace(last_coords[0], new_coords[0], num_frames + 2)
        y_points = np.linspace(last_coords[1], new_coords[1], num_frames + 2)
        return [(int(x), int(y)) for x, y in zip(x_points[1:-1], y_points[1:-1])]

    def show_repaired_frames(self, frames, coords_list, save_output=False, video_writer=None, fps=30):
        """
        Iterates over the buffered frames, draws the orange bounding indicators, 
        and flushes them into the video writer or live preview stream in chronological order.
        """
        live_delay = max(1, int(1000 / fps))
        
        for frame, coords in zip(frames, coords_list):
            if coords is not None:
                cx, cy = int(coords[0]), int(coords[1])
                cv2.circle(frame, (cx, cy), 12, (0, 140, 255), 5)
                cv2.putText(frame, "Offline", (cx - 25, cy - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            if save_output and video_writer is not None:
                video_writer.write(frame)
            else:
                cv2.imshow("TagYourMoves - Live Local Inference", frame)
                cv2.waitKey(live_delay)