from src.objects.Ball import Ball
from src.objects.Player import Player
from src.objects.Rim import Rim

class FrameSnapshot:
    def __init__(self, frame_index):
        self.frame_index = frame_index
        self.ball = None
        self.players = []   
        self.rims = []   

    def set_ball(self, x_center, y_center, x1, y1, x2, y2):
        self.ball = Ball(x_center, y_center, x1, y1, x2, y2)

    def add_player(self, x1, y1, x2, y2, player_id):
        self.players.append(Player(x1, y1, x2, y2, player_id))

    def add_rim(self, x1, y1, x2, y2):
        self.rims.append(Rim(x1, y1, x2, y2))

    def is_ball_missing(self):
        return self.ball is None