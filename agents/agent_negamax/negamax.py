import numpy as np
from agents.common import *
from typing import Callable, Optional, Tuple

def valid_columns(board: np.ndarray):
    return np.where(board[-1,:] == NO_PLAYER)[0]

def heuristic(board: np.ndarray, player: BoardPiece):
    if check_end_state(board, player) == GameState.IS_WIN:
        return 1000000
    else:
        return 0

def is_terminal_board(board: np.ndarray, player: BoardPiece):
    if check_end_state(board, player) != GameState.STILL_PLAYING:
        return True
    else: 
        return False

# def negamax(board: np.ndarray, player: BoardPiece, depth: np.int8):
#     if depth == 0 or is_terminal_board(board, player):
#         return (-1, heuristic(board, player))
#     value = -np.inf
#     for move in valid_columns(board):
#         child = apply_player_action(board, move, player)
#         new_value = np.max(value, −1*negamax(child, depth − 1, -1*player)[1])
#         if new_value > value:
#             best_action = move
#     return best_action, value

def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:

    action = alphabeta(board, 3, player)[0]

    return action, saved_state