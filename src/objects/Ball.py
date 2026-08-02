class Ball:
    def __init__(self, x_center, y_center, x1, y1, x2, y2):
        self.x_center = x_center
        self.y_center = y_center
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def exists(self):
        return self.x_center is not None and self.y_center is not None