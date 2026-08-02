class Rim:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def ball_touch(self, ball):
        # Check if the ball's center is within the rim's bounding box
        return self.x1 <= ball.x_center <= self.x2 and self.y1 <= ball.y_center <= self.y2