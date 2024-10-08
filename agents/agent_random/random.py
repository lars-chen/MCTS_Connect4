import numpy as np
from agents.common import *
from typing import Callable, Optional, Tuple

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Runs the random actions.
    """
    valid_columns = np.where(board[-1,:] == NO_PLAYER)[0]
    action = np.random.choice(valid_columns)

    return action, saved_state