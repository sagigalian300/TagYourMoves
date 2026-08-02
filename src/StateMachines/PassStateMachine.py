from src.geometry import is_point_in_box

class PassStateMachine:
    def __init__(self, fps):
        self.state = 'IDLE'  # Initial state
        self.detect_goal = "PASS"

        self.player1_id = None
        self.player2_id = None

        self.player1_touch_frame = None
        self.player2_touch_frame = None

        self.fps = fps

    def _is_ball_on_player(self, ball, players):
        for player in players:
            if player.hold_ball(ball):
                return player
        return None
    
    def update(self, frame_snapshot):
        players = frame_snapshot.players
        ball = frame_snapshot.ball

        match self.state:
            case 'IDLE':
                player = self._is_ball_on_player(ball, players)
                if player:
                    self.state = 'BALL_ON_PLAYER1'
                    self.player1_id = player.id
                    self.player1_touch_frame = frame_snapshot.frame_index

            case 'BALL_ON_PLAYER1':
                if not self._is_ball_on_player(ball, players):
                    self.state = 'BALL_IN_AIR'
                else:
                    self.player1_touch_frame = frame_snapshot.frame_index

            case 'BALL_IN_AIR':
                player = self._is_ball_on_player(ball, players)
                if player and player.id != self.player1_id:
                    self.player2_id = player.id
                    self.state = 'BALL_ON_PLAYER2'
                    self.player2_touch_frame = frame_snapshot.frame_index

            case 'BALL_ON_PLAYER2':
                print(self)
                self.reset()
                return True

        return False

    def reset(self):
        self.state = 'IDLE'
        self.player1_id = None
        self.player2_id = None
        self.player1_touch_frame = None
        self.player2_touch_frame = None

    def __str__(self):
        return f"--------\nPass Detected:\nPlayer 1 ID: {self.player1_id}, Player 2 ID: {self.player2_id}\nPlayer 1 touch frame: {self.player1_touch_frame}, in sec: {self.player1_touch_frame / self.fps}\nPlayer 2 touch frame: {self.player2_touch_frame}, in sec: {self.player2_touch_frame / self.fps}\n--------"