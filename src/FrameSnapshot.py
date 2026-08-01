from dataclasses import dataclass, field

@dataclass
class FrameSnapshot:
    frame_index: int
    ball_center: tuple = None
    players_boxes: list = field(default_factory=list)
    rims_boxes: list = field(default_factory=list)

    @property
    def is_ball_missing(self) -> bool:
        return self.ball_center is None