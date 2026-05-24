import cv2
import os

class VideoManager:
    def __init__(self, input_path, output_path):
        """
        Utility class to encapsulate OpenCV video I/O operations and metadata management.
        """
        self.input_path = input_path
        self.output_path = output_path
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found at: {input_path}")
            
        self.cap = cv2.VideoCapture(input_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

    def read_frame(self):
        """
        Reads the next video frame.
        :return: Tuple (ret, frame)
        """
        return self.cap.read()

    def write_frame(self, frame):
        """
        Writes the processed frame into the output video container.
        """
        self.writer.write(frame)

    def release(self):
        """
        Closes and releases video capture and writer resources safely.
        """
        self.cap.release()
        self.writer.release()
        
    def get_dimensions(self):
        return self.width, self.height
