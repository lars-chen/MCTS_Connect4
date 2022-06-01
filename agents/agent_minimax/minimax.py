import numpy as np
from agents.common import *
from typing import Callable, Optional, Tuple

def valid_columns(board: np.ndarray):
    return np.where(board[-1,:] == NO_PLAYER)[0]

def heuristic(board: np.ndarray, player: BoardPiece, maximizingPlayer = True):
    if check_end_state(board, player) == GameState.IS_WIN:
        return 1000000000
    elif check_end_state(board, -1*player) == GameState.IS_WIN:
        return -1000000000
    else:
        return 0

def is_terminal_board(board: np.ndarray, player: BoardPiece):
    if check_end_state(board, player) != GameState.STILL_PLAYING:
        return True
    else: 
        return False

def alphabeta(board: np.ndarray, player: BoardPiece, depth: np.int8, maximizingPlayer = True, alpha = np.NINF, beta = np.PINF):
    valid_actions = valid_columns(board)
    if depth == 0 or is_terminal_board(board, player):
         return -1, heuristic(board, player)
    best_action = 0
    if maximizingPlayer:
        value = np.NINF
        for move in valid_actions:
            child = apply_player_action(board, move, player)
            print(value)
            new_value = alphabeta(child, player, depth - 1, False, alpha, beta)[1]
            if new_value > value:
                best_action = move
            if new_value >= beta:
                break 
            alpha = max(alpha, value)
        return best_action, value
    else:
        value = np.PINF
        for move in valid_actions:
            child = apply_player_action(board, move, player)
            new_value = alphabeta(child, player, depth - 1, True, alpha, beta)[1]
            if new_value < value:
                best_action = move
            if new_value <= alpha:
                break 
            beta = min(beta, value)
        return best_action, value

def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    depth = 4
    action = alphabeta(board, player, depth, True)[0]
    print('action:', action)
    return action, saved_state