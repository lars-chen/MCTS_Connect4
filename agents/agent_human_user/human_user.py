import numpy as np

from agents.common import BoardPiece, PlayerAction, SavedState, NO_PLAYER
from typing import Optional, Callable, Tuple


def user_move(board: np.ndarray,
              _player: BoardPiece,
              saved_state: Optional[SavedState]) -> Tuple[PlayerAction, SavedState]:
    is_valid_move = False
    while not is_valid_move:
        input_move_string = query_user(input)
        try:
            is_valid_move = handle_illegal_moves(board, input_move_string)
        except TypeError:
            print('Not the right format, try an integer.')
        except IndexError:
            print('Selected integer is not in the range of possible columns (0 - 6).')
        except ValueError:
            print('Selected column is full.')
    input_move_integer = PlayerAction(input_move_string)
    return input_move_integer, saved_state


def query_user(prompt_function: Callable):
    usr_input = prompt_function("Column? ")
    return usr_input


def handle_illegal_moves(board: np.ndarray, column: PlayerAction):
    try:
        column = PlayerAction(column)
    except:
        raise TypeError

    is_in_range = PlayerAction(0) <= column <= PlayerAction(6)
    if not is_in_range:
        raise IndexError

    is_open = board[-1, column] == NO_PLAYER
    if not is_open:
        raise ValueError

    return True