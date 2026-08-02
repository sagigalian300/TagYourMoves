from src.geometry import is_point_in_box

class ShotStateMachine:
    def __init__(self):
        self.state = 'IDLE'  # Initial state

    def _is_ball_on_player(self, ball, players):
        for player_box in players:
            if is_point_in_box(ball, player_box):
                return True
        return False

    def _is_ball_near_rim(self, ball, rims):
        for rim_box in rims:
            if is_point_in_box(ball, rim_box):
                return True
        return False
    
    def update(self, frame_snapshot):
        players = frame_snapshot.players_boxes
        ball = frame_snapshot.ball_center
        rims = frame_snapshot.rims_boxes

        match self.state:
            case 'IDLE':
                for player_box in players:
                    if is_point_in_box(ball, player_box):
                        self.state = 'BALL_ON_PLAYER'

            case 'BALL_ON_PLAYER':
                if not self._is_ball_on_player(ball, players):
                    self.state = 'BALL_IN_AIR'

            case 'BALL_IN_AIR':
                if self._is_ball_near_rim(ball, rims):
                    self.state = 'BALL_NEAR_RIM'

                elif self._is_ball_on_player(ball, players):
                    self.state = 'BALL_ON_PLAYER'

            case 'BALL_NEAR_RIM':
                self.state = 'IDLE'
                return True

        return False
     