from enum import Enum
import numpy as np
from scipy import signal
from typing import Callable, Optional, Tuple

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(
    0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(
    1
)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(
    -1
)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece
rows = BoardPiece(6)
columns = BoardPiece(7)

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint(f'X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = 2
    STILL_PLAYING = 0

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.full((rows, columns), NO_PLAYER, dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    board = board[::-1] 
    pp_board = f"|==============|\n"  # add top border of board
    for row in range(board.shape[0]):
        pp_board += f"|" + np.array2string(board[row, :]).replace(
            '[', '').replace(']', '').replace('-1', PLAYER2_PRINT) + " |\n"  # replace player values with strings: -1 value allows use of convolutions

        if np.any(board[row, :] == PLAYER2):  # handle both -1 and 1 cases, since -1 is two characters
            pp_board = pp_board.replace(' 0', NO_PLAYER_PRINT).replace(
                ' 1', PLAYER1_PRINT)
        else:
            pp_board = pp_board.replace('0', NO_PLAYER_PRINT).replace(
                '1', PLAYER1_PRINT)

    pp_board += f"|==============|\n|0 1 2 3 4 5 6 |"  # add bottom border
    return pp_board


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    board = pp_board[18:-33].replace('|\n|', '').replace('|', '')[0:-1:2]  # remove borders of pretty print board
    board = np.reshape([
        PLAYER1 if board[i] == PLAYER1_PRINT else  # retrieve player values from string board
        PLAYER2 if board[i] == PLAYER2_PRINT else NO_PLAYER
        for i in range(len(board))
    ], (rows, columns))
    return board[::-1]


def apply_player_action(board: np.ndarray, action: PlayerAction,
                        player: BoardPiece) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. Raises a ValueError
    if action is not a legal move. If it is a legal move, the modified version of the
    board is returned and the original board should remain unchanged (i.e., either set
    back or copied beforehand).
    """

    if action > columns - 1 or action < 0:
        raise ValueError('Action outside of board.')

    if ~np.any(board[:, action] == NO_PLAYER):
        raise ValueError('Column is already full.')

    modified_board = board.copy()
    for i in range(rows):
        if modified_board[:, action][i] == NO_PLAYER:  # find first non-filled space in column
            modified_board[:, action][i] = player
            break

    return modified_board


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    vertical_kernel = np.ones((4, 1), dtype=BoardPiece)
    horizontal_kernel = np.ones((1, 4), dtype=BoardPiece)
    diagonal_kernel = np.eye(4, dtype=BoardPiece)
    off_diagonal_kernel = np.eye(4, dtype=BoardPiece)[::-1]

    is_win = False
    if np.any(signal.convolve2d(board, vertical_kernel, 'same') == player * 4):
        is_win = True
    elif np.any(signal.convolve2d(board, horizontal_kernel, 'same') == player * 4):
        is_win = True
    elif np.any(signal.convolve2d(board, diagonal_kernel, 'same') == player * 4):
        is_win = True
    elif np.any(signal.convolve2d(board, off_diagonal_kernel, 'same') == player * 4):
        is_win = True
    return is_win


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player):
        return GameState.IS_WIN

    elif np.all(board != NO_PLAYER):
        return GameState.IS_DRAW

    else:
        return GameState.STILL_PLAYING


def is_terminal_board(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True only if the game is over at the current state.
    """
    if check_end_state(board, player) != GameState.STILL_PLAYING:
        return True
    else:
        return False