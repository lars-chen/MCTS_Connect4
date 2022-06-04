import numpy as np
from agents.common import PLAYER1, PLAYER2, BoardPiece, NO_PLAYER, GameState, initialize_game_state, pretty_print_board, string_to_board


def test_initialize_game_state():
    #from agents.common import initialize_game_state

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
    board[5, :] = PLAYER1
    board[:, 4] = PLAYER2
    pp_board = pretty_print_board(board)

    assert np.any((board == string_to_board(pp_board)) == True)


def test_apply_player_action():
    from agents.common import apply_player_action

    board = initialize_game_state()
    board = apply_player_action(board, 0, PLAYER1)
    assert board[0, 0] == PLAYER1

    for i in range(5):
        board = apply_player_action(board, 0, PLAYER1)
    assert np.all(board[:, 0])

    board = apply_player_action


def test_connected_four():
    from agents.common import connected_four

    # test horizontal win for both players
    board = initialize_game_state()
    board[-1, :] = PLAYER1
    assert connected_four(board, PLAYER1)

    board[-1, :] = PLAYER2
    assert connected_four(board, PLAYER2)

    # test diagonal win
    board = initialize_game_state()
    board[np.where(
        np.diag([NO_PLAYER, NO_PLAYER, PLAYER1, PLAYER1, PLAYER1,
                 PLAYER1]))] = 1
    assert connected_four(board, PLAYER1)

    # test diagonal win
    board = initialize_game_state()
    board[np.where(
        np.diag([NO_PLAYER, NO_PLAYER, PLAYER1, PLAYER1, PLAYER1,
                 PLAYER1])[::-1])] = 1
    assert connected_four(board, PLAYER1)


def test_check_end_state():
    from agents.common import check_end_state

    # test if game is still in playing state
    board = initialize_game_state()
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

    # test win state
    board[np.where(
        np.diag([NO_PLAYER, NO_PLAYER, PLAYER1, PLAYER1, PLAYER1,
                 PLAYER1])[::-1])] = 1
    assert check_end_state(board, PLAYER1) == GameState.IS_WIN

    board = initialize_game_state()
