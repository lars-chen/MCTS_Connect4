import numpy as np
from agents.common import PLAYER1, PLAYER2, BoardPiece, NO_PLAYER, GameState, apply_player_action, connected_four, initialize_game_state, pretty_print_board, string_to_board
import pytest


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    from agents.common import pretty_print_board

    board = initialize_game_state()
    board[5, :] = PLAYER1
    board[:, 4] = PLAYER2

    pp_board = pretty_print_board(board)
    print(pp_board)
    assert type(pp_board) == str
    assert len(pp_board) == 152


def test_string_to_board():
    from agents.common import string_to_board

    board = initialize_game_state()
    board = apply_player_action(board, 0, PLAYER1)
    board = apply_player_action(board, 0, PLAYER2)
    pp_board = pretty_print_board(board)

    assert np.all((board == string_to_board(pp_board)) == True)


def test_apply_player_action_player_2():
    from agents.common import apply_player_action

    board = initialize_game_state()
    board = apply_player_action(board, 0, PLAYER2)
    assert board[0, 0] == PLAYER2


def test_apply_player_action_stacking():
    from agents.common import apply_player_action

    board = initialize_game_state()
    board = apply_player_action(board, 0, PLAYER1)
    board = apply_player_action(board, 0, PLAYER2)
    assert board[1, 0] == PLAYER2


def test_apply_player_action_player_error_2():
    from agents.common import apply_player_action
    board = initialize_game_state()
    for _ in range(6):
        board = apply_player_action(board, 0, PLAYER1)
    pytest.raises(ValueError, apply_player_action, board, 0, PLAYER1)


def test_apply_player_action_player_error_1():
    from agents.common import apply_player_action
    board = initialize_game_state()

    pytest.raises(ValueError, apply_player_action, board, -1, PLAYER1)
    pytest.raises(ValueError, apply_player_action, board, 7, PLAYER1)


def test_connected_four_horizontal():
    pretty_horizontal_board = (
    "|==============|\n"
    "|      O       |\n"
    "|      O       |\n"
    "|      O       |\n"
    "|      X       |\n"
    "|      O       |\n"
    "|  X X X X     |\n"
    "|==============|\n"
    "|0 1 2 3 4 5 6 |"
    )
    is_win = connected_four(string_to_board(pretty_horizontal_board), PLAYER1)

    assert is_win == True

def test_connected_four_vertical():
    pretty_vertical_board = (
    "|==============|\n"
    "|      O       |\n"
    "|      O       |\n"
    "|      O       |\n"
    "|      O       |\n"
    "|      X       |\n"
    "|  X X X       |\n"
    "|==============|\n"
    "|0 1 2 3 4 5 6 |"
    )
    is_win = connected_four(string_to_board(pretty_vertical_board), PLAYER2)

    assert is_win == True

def test_connected_four_diagonal():
    pretty_diagonal_board = (
    "|==============|\n"
    "|      O       |\n"
    "|      O       |\n"
    "|      O   X   |\n"
    "|      X X X   |\n"
    "|  X O X O O   |\n"
    "|  X X X O X   |\n"
    "|==============|\n"
    "|0 1 2 3 4 5 6 |"
    )
    is_win = connected_four(string_to_board(pretty_diagonal_board), PLAYER1)

    assert is_win == True

def test_connected_four_off_diagonal():
    pretty_off_diagonal_board = (
    "|==============|\n"
    "|      O       |\n"
    "|      O       |\n"
    "|    X O       |\n"
    "|    X X       |\n"
    "|    O O X     |\n"
    "|    O X X X   |\n"
    "|==============|\n"
    "|0 1 2 3 4 5 6 |"
    )
    is_win = connected_four(string_to_board(pretty_off_diagonal_board), PLAYER1)

    assert is_win == True


def test_check_end_state_win():
    from agents.common import check_end_state

    # test win state
    pretty_off_diagonal_board = (
    "|==============|\n"
    "|      X       |\n"
    "|      O       |\n"
    "|    O O       |\n"
    "|    X O       |\n"
    "|    O O O     |\n"
    "|    O X X O   |\n"
    "|==============|\n"
    "|0 1 2 3 4 5 6 |"
    )
    assert check_end_state(string_to_board(pretty_off_diagonal_board), PLAYER2) == GameState.IS_WIN

def test_check_end_state_still_playing():
    from agents.common import check_end_state

    # test if game is still in playing state
    board = initialize_game_state()
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING
    assert check_end_state(board, PLAYER2) == GameState.STILL_PLAYING

def test_check_end_state_draw():
    from agents.common import check_end_state

    # test win state
    pretty_off_full_board = (
    "|==============|\n"
    "|O X O X O X O |\n"
    "|O X O X O X O |\n"
    "|X O X O X O X |\n"
    "|X O X O X O X |\n"
    "|O X O X O X O |\n"
    "|O X O X O X O |\n"
    "|==============|\n"
    "|0 1 2 3 4 5 6 |"
    )
    board = string_to_board(pretty_off_full_board)
    assert check_end_state(board, PLAYER2) == GameState.IS_DRAW
    assert check_end_state(board, PLAYER1) == GameState.IS_DRAW