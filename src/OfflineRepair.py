import numpy as np

class OfflineRepair:
    def __init__(self):
        """
        Initializes the repair class. 
        No instance variables are needed here, as everything is calculated on the fly.
        """
        pass

    def get_repaired_coords(self, start_coords, end_coords, num_frames):
        """
        Calculates a linear interpolation (straight line) between two points.
        
        :param start_coords: (X, Y) The last known position of the ball.
        :param end_coords: (X, Y) The new anchor position where the ball reappeared.
        :param num_frames: The number of missing frames in between (length of the buffer).
        :return: A list of (X, Y) coordinates for the missing frames.
        """
        if num_frames <= 0 or start_coords is None or end_coords is None:
            return []
            
        # Create an array of evenly spaced points.
        # We add 2 to the number of frames to include both the start and end points in the calculation.
        x_points = np.linspace(start_coords[0], end_coords[0], num_frames + 2)
        y_points = np.linspace(start_coords[1], end_coords[1], num_frames + 2)
        
        # Slice out the start and end points (since we already have them) 
        # and return only the middle points as a list of (X, Y) tuples.
        return [(float(x), float(y)) for x, y in zip(x_points[1:-1], y_points[1:-1])]