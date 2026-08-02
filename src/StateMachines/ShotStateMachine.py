class ShotStateMachine:
    def __init__(self, fps):
        self.state = 'IDLE'  # Initial state
        self.detect_goal = "SHOT"

        self.on_player = None
        self.on_rim = None

        self.fps = fps
        
    def _is_ball_on_player(self, ball, players):
        for player in players:
            if player.hold_ball(ball):
                return True
        return False

    def _is_ball_near_rim(self, ball, rims):
        for rim in rims:
            if rim.ball_touch(ball):
                return True
        return False
    
    def update(self, frame_snapshot):
        players = frame_snapshot.players
        ball = frame_snapshot.ball
        rims = frame_snapshot.rims

        if ball is None:
            return False
        
        match self.state:
            case 'IDLE':
                if self._is_ball_on_player(ball, players):
                    self.state = 'BALL_ON_PLAYER'
                    if not self.on_player:
                        self.on_player = frame_snapshot.frame_index

            case 'BALL_ON_PLAYER':
                if not self._is_ball_on_player(ball, players):
                    self.state = 'BALL_IN_AIR'
                    self.on_player = frame_snapshot.frame_index
                    
            case 'BALL_IN_AIR':
                if self._is_ball_near_rim(ball, rims):
                    self.state = 'BALL_NEAR_RIM'

                elif self._is_ball_on_player(ball, players):
                    self.state = 'BALL_ON_PLAYER'

            case 'BALL_NEAR_RIM':
                self.on_rim = frame_snapshot.frame_index
                print(self)
                self.reset()
                return True

        return False

    def reset(self):
        self.state = 'IDLE'
        self.on_player = None
        self.on_rim = None

    def __str__(self):
        return f"--------\nShot Detected:\nPlayer first hold: in frame: {self.on_player}, in sec: {self.on_player / self.fps}\nOn rim: in frame: {self.on_rim}, in sec: {self.on_rim / self.fps}\n--------"
     