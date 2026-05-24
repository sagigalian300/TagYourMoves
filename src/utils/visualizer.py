import cv2

class Visualizer:
    def __init__(self):
        """
        Utility class for drawing detection anchors and dynamic HUD overlays on frames.
        """
        # BGR Colors
        self.ball_color = (0, 165, 255)  # Orange
        self.rim_color = (0, 0, 255)     # Red
        self.hud_color = (0, 128, 255)    # Orange-Alert Banner

    def draw_predictions(self, frame, ball_box, rim_box):
        """
        Draws precise bounding boxes for the ball and the rim if they are present.
        """
        # Draw Ball Box if available
        if ball_box is not None:
            x1, y1, x2, y2 = map(int, ball_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.ball_color, 2)
            cv2.putText(frame, "Ball", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ball_color, 2)

        # Draw Rim Box if available
        if rim_box is not None:
            x1, y1, x2, y2 = map(int, rim_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.rim_color, 2)
            cv2.putText(frame, "Rim", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.rim_color, 2)

    def draw_shot_alert(self, frame, probability, width, height):
        """
        Flashes the upper HUD banner and screen border when a shot is detected.
        """
        # Top HUD Banner notification box
        cv2.rectangle(frame, (30, 30), (450, 100), self.hud_color, -1)
        cv2.putText(frame, f"SHOT DETECTED ({probability*100:.1f}%)", (45, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Screen bounding notification border
        cv2.rectangle(frame, (0, 0), (width, height), self.hud_color, 8)
