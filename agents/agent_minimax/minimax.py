disable_jit = True
if disable_jit:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
from numba import njit
from agents.common import *
from typing import Callable, Optional, Tuple


def sorted_valid_columns(board: np.ndarray) -> np.ndarray:
    """
    Returns an array of columns that may be played into that are sorted from middle to outside.
    Sorting from inside to outside makes alpha-beta pruning more efficient and the computational 
    time for the agent more even. 
    """
    valid_columns = np.where(board[-1, :] == NO_PLAYER)[
        0]  # check the top row of the board to see where there are open spaces
    mid = int(np.ceil(len(valid_columns) / 2))
    sorted_valid = np.empty(len(valid_columns))
    sorted_valid[0::2] = np.flip(
        valid_columns[:mid]
    )  # construct array that is sequentially sorted from inside to outside
    sorted_valid[1::2] = valid_columns[mid:]

    return sorted_valid.astype(np.int8)


@njit()
def connected_n(
    board: np.ndarray,
    player: BoardPiece,
    n: BoardPiece,
    num_rows=BoardPiece(6),
    num_columns=BoardPiece(7)) -> np.int8:
    """
    Returns number of unique possibilities of 4 spaces in a row that could become four in a row which currently have
    n of the current player's pieces. This function breaks down the game board into all possible vertical, horizontal, 
    diagonal and off-diagonal runs of four spaces. Given a number n and the current board, this function counts 
    the runs that has the player's pieces n times and the other positions have no player. 
    """
    count_n = 0

    # initialize connect 4 possibilities
    horizontal_kernel = np.ones((1, 4), dtype=np.int8)
    vertical_kernel = np.ones((4, 1), dtype=np.int8)
    diagonal_kernel = np.eye(4, dtype=np.int8)
    off_diagonal_kernel = np.eye(4, dtype=np.int8)[::-1]
    kernels = [
        vertical_kernel, horizontal_kernel, off_diagonal_kernel,
        diagonal_kernel
    ]

    for kernel in kernels:
        kernel_width = kernel.shape[1]
        kernel_height = kernel.shape[0]
        for i in range(num_rows - kernel_height + 1):  # loop through rows such that kernel fits in the board
            for j in range(num_columns - kernel_width + 1):  # loop through columns such that kernel fits
                sample_array = board[i:i + kernel_height, j:j + kernel_width][
                    np.where(kernel)]  # get all possible runs of 4
                if (np.sum(
                        sample_array[sample_array == player]
                ) == player * n) and (
                        len(sample_array[sample_array == NO_PLAYER]) == 4 - n
                ):  # check if player is in sample n times and connect 4 still possible
                    count_n += 1
    return count_n


def heuristic(board: np.ndarray, player: BoardPiece,
              maximizingPlayer: bool) -> np.int8:
    """
    Returns heuristic score of the current gamestate depending on if the player is maximizing or 
    minimizing. The maximizing and minimizing players's win conditions are prioritized with the highest 
    absolute score. Next, the heuristic prioritizes a high number of streaks of 3 and subsequently streaks 
    of 2 given by the connected_n function.
    """
    score = 0
    other_player = PLAYER1 if player == PLAYER2 else PLAYER2

    if check_end_state(board, player) == GameState.IS_WIN:
        score = 1e15
    else:  # if win condition exists, don't check other possibilities
        connected_threes = connected_n(board, player, 3)
        if connected_threes > 0:
            score += 2e8 * connected_threes  #  prioritize connected 3 streaks with multiple options for connected 4
        else:
            score += 1e5 * connected_n(board, player, 2)

    if check_end_state(board, other_player) == GameState.IS_WIN:
        score = -1e14
    else: # if lose condition exists, don't check other possibilities
        connected_threes_other = connected_n(board, other_player, 3)
        if connected_threes_other > 0:
            score -= 1e8 * connected_threes_other
        else:
            score -= 1e5 * connected_n(board, other_player, 2)

    return score if maximizingPlayer else -score


def is_terminal_board(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True only if the game is over at the current state.
    """
    if check_end_state(board, player) != GameState.STILL_PLAYING:
        return True
    else:
        return False

@njit()
def alphabeta(board: np.ndarray,
              player: BoardPiece,
              depth: np.int8,
              maximizingPlayer=True,
              alpha=np.NINF,
              beta=np.PINF) -> np.int8:
    """
    Returns the best possible action as defined by a heuristic function and checks moves at
    inner columns first and outside moves last. The minimax agent employs alpha-beta pruning 
    for efficiency of search.
    """

    valid_actions = sorted_valid_columns(board)
    best_action = None
    other_player = PLAYER1 if player == PLAYER2 else PLAYER2  # establish player and opponent

    if (depth == 0) or (is_terminal_board(board, player)):
        return -1, heuristic(board, player, maximizingPlayer)
    if maximizingPlayer:
        value = np.NINF
        for move in valid_actions:
            child = apply_player_action(board, move, player)
            new_value = alphabeta(child, other_player, depth - 1, False, alpha,
                                  beta)[1]
            if new_value > value:
                best_action = move
                value = new_value
            if value >= beta:  # beta cutoff
                break
            alpha = max(alpha, value)
        return best_action, value
    else:
        value = np.PINF
        for move in valid_actions:
            child = apply_player_action(board, move, player)
            new_value = alphabeta(child, other_player, depth - 1, True, alpha,
                                  beta)[1]
            if new_value < value:
                best_action = move
                value = new_value
            if value <= alpha:  # alpha cutoff
                break
            beta = min(beta, value)
        return best_action, value


def generate_move_minimax(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[SavedState],
    depth=np.int8(4)
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Runs the minimax algorithm and returns best action.
    """
    action, value = alphabeta(board, player, depth, True)
    return action, saved_state