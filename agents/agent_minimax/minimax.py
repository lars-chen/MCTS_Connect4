import numpy as np
from agents.common import *
from typing import Callable, Optional, Tuple

def valid_columns(board: np.ndarray):
    """
    Returns an array of columns that may be played into.
    """
    return np.where(board[-1,:] == NO_PLAYER)[0]  # check the top row of the board to see where there are open spaces

def connected_n(board: np.ndarray, player: BoardPiece):

    return

def heuristic(board: np.ndarray, player: BoardPiece, maximizingPlayer: bool):
    """
    Returns heuristic score of the current gamestate depending on if the player is maximizing or minimizing.
    """
    heur_value = 0
    other_player = PLAYER1 if player == PLAYER2 else PLAYER2

    if check_end_state(board, player) == GameState.IS_WIN:
        heur_value += 1000000000

    if check_end_state(board, other_player) == GameState.IS_WIN:
        heur_value -= 100000000

    if not maximizingPlayer:
        heur_value = -heur_value
    return heur_value

def is_terminal_board(board: np.ndarray, player: BoardPiece):
    """
    Returns True only if the game is over at the current state.
    """
    if check_end_state(board, player) != GameState.STILL_PLAYING:
        return True
    else: 
        return False

def alphabeta(board: np.ndarray, player: BoardPiece, depth: np.int8, maximizingPlayer = True, alpha = np.NINF, beta = np.PINF):
    valid_actions = valid_columns(board)
    best_action = None
    other_player = PLAYER1 if player == PLAYER2 else PLAYER2

    if (depth == 0) or (is_terminal_board(board, player)):
         return -1, heuristic(board, player, maximizingPlayer)
    if maximizingPlayer:
        value = np.NINF
        for move in valid_actions:
            child = apply_player_action(board, move, player)
            new_value = alphabeta(child, other_player, depth - 1, False, alpha, beta)[1]
            if new_value > value:
                best_action = move
                value = new_value
            if value >= beta:
                break 
            alpha = max(alpha, value)
        return best_action, value
    else:
        value = np.PINF
        for move in valid_actions:
            child = apply_player_action(board, move, player)
            new_value = alphabeta(child, other_player, depth - 1, True, alpha, beta)[1]
            if new_value < value:
                best_action = move
                value = new_value
            if value <= alpha:
                break 
            beta = min(beta, value)
        return best_action, value

def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    depth = 4
    action, value = alphabeta(board, player, depth, True)
    print('action:', action, 'value: ', value)
    return action, saved_state