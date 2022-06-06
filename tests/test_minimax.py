from mimetypes import init
from agents.common import *
import numpy as np
from agents.agent_minimax.minimax import alphabeta, generate_move_minimax, heuristic


def test_sorted_valid_columns_on_intitial_state():
    from agents.agent_minimax.minimax import sorted_valid_columns

    board = initialize_game_state()
    assert (sorted_valid_columns(board) == np.array([3, 4, 2, 5, 1, 6, 0],
                                                    dtype=np.int8)).all()


def test_sorted_valid_columns_full_column():
    from agents.agent_minimax.minimax import sorted_valid_columns

    board = initialize_game_state()
    for _ in range(6):
        board = apply_player_action(board, 0, PLAYER1)
    assert (sorted_valid_columns(board) == np.array([3, 4, 2, 5, 1, 6],
                                                    dtype=np.int8)).all()


def test_connected_n3_double_threat():
    from agents.agent_minimax.minimax import connected_n
    pretty_board = ("|==============|\n"
                             "|              |\n"
                             "|              |\n"
                             "|              |\n"
                             "|              |\n"
                             "|              |\n"
                             "|    X X X     |\n"
                             "|==============|\n"
                             "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    assert connected_n(board, PLAYER1, n=3) == 2


def test_connected_n3_horizontal_single_threat():
    from agents.agent_minimax.minimax import connected_n
    pretty_board = ("|==============|\n"
                             "|              |\n"
                             "|              |\n"
                             "|              |\n"
                             "|              |\n"
                             "|              |\n"
                             "|    X   X X   |\n"
                             "|==============|\n"
                             "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    assert connected_n(board, PLAYER1, n=3) == 1


def test_connected_n3_diagonal_single_threat():
    from agents.agent_minimax.minimax import connected_n
    pretty_board = ("|==============|\n"
                             "|              |\n"
                             "|              |\n"
                             "|              |\n"
                             "|      X       |\n"
                             "|    X O       |\n"
                             "|  X O O       |\n"
                             "|==============|\n"
                             "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    assert connected_n(board, PLAYER1, n=3) == 1


def test_connected_n3_diagonal_double_threat():
    from agents.agent_minimax.minimax import connected_n
    pretty_board = ("|==============|\n"
                             "|              |\n"
                             "|              |\n"
                             "|        X     |\n"
                             "|      X O     |\n"
                             "|    X O O     |\n"
                             "|    O O X     |\n"
                             "|==============|\n"
                             "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    assert connected_n(board, PLAYER1, n=3) == 2


def test_connected_n3_off_diagonal_double_threat():
    from agents.agent_minimax.minimax import connected_n
    pretty_board = ("|==============|\n"
                             "|              |\n"
                             "|              |\n"
                             "|    O         |\n"
                             "|    X O       |\n"
                             "|    X X O     |\n"
                             "|  X O O X     |\n"
                             "|==============|\n"
                             "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    assert connected_n(board, PLAYER2, n=3) == 2


def test_connected_n3_vertical_single_threat():
    from agents.agent_minimax.minimax import connected_n
    pretty_board = ("|==============|\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|    O X       |\n"
                               "|    O X       |\n"
                               "|    O X       |\n"
                               "|==============|\n"
                               "|0 1 2 3 4 5 6 |")

    board = string_to_board(pretty_board)
    assert connected_n(board, PLAYER1, n=3) == 1
    assert connected_n(board, PLAYER2, n=3) == 1


def test_connected_n2_triple_threat():
    from agents.agent_minimax.minimax import connected_n
    pretty_board = ("|==============|\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|    O O       |\n"
                               "|==============|\n"
                               "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    assert connected_n(board, PLAYER2, n=2) == 3


def test_connected_n2_double_threat():
    from agents.agent_minimax.minimax import connected_n
    pretty_board = ("|==============|\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|        X     |\n"
                               "|    O   O   X |\n"
                               "|==============|\n"
                               "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    assert connected_n(board, PLAYER2, n=2) == 2


def test_connected_n2_single_threat():
    from agents.agent_minimax.minimax import connected_n
    pretty_board = ("|==============|\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|    O     O   |\n"
                               "|==============|\n"
                               "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    assert connected_n(board, PLAYER2, n=2) == 1


def test_heuristic_initial_game_state():
    from agents.agent_minimax.minimax import heuristic

    board = initialize_game_state()
    assert heuristic(board, PLAYER1, True) == 0


def test_heuristic_n2_single_threat():
    from agents.agent_minimax.minimax import heuristic
    pretty_board = ("|==============|\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|              |\n"
                               "|    X         |\n"
                               "|    O     O X |\n"
                               "|==============|\n"
                               "|0 1 2 3 4 5 6 |")

    board = string_to_board(pretty_board)
    assert heuristic(board, PLAYER2, True) == 1e5


def test_heuristic_n2_double_threat():
    from agents.agent_minimax.minimax import heuristic
    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|    X         |\n"
                    "|    O   O   X |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")

    board = string_to_board(pretty_board)
    assert heuristic(board, PLAYER2, True) == 2e5


def test_heuristic_n3_double_threat_single_streak():
    from agents.agent_minimax.minimax import heuristic
    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|    X         |\n"
                    "|    O O O   X |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")

    board = string_to_board(pretty_board)
    assert heuristic(board, PLAYER2, True) == 4e8


def test_heuristic_n3_double_threat_double_streak():
    from agents.agent_minimax.minimax import heuristic
    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|        O     |\n"
                    "|    X   O     |\n"
                    "|    O O O X X |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")

    board = string_to_board(pretty_board)
    assert heuristic(board, PLAYER2, True) == 4e8


def test_heuristic_win_lose():
    from agents.agent_minimax.minimax import heuristic
    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|        O     |\n"
                    "|        O     |\n"
                    "|X X X X O     |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")

    board = string_to_board(pretty_board)
    assert heuristic(board, PLAYER1, True) == 1e15 - 1e8
    assert heuristic(board, PLAYER2, True) == -1e14


def test_alphabeta_initial_move_on_center():
    from agents.agent_minimax.minimax import alphabeta

    board = initialize_game_state()
    action = alphabeta(board, PLAYER1, 4)[0]
    assert action == 3


def test_alphabeta_depth_one_chooses_win():
    from agents.agent_minimax.minimax import alphabeta

    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|O   X X X O O |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    action, value = alphabeta(board, PLAYER1, 1, True, np.NINF, np.PINF)
    
    assert action == 1

def test_alphabeta_depth_one_prevents_loss():
    from agents.agent_minimax.minimax import alphabeta

    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|  O   O X O   |\n"
                    "|  X   O O X   |\n"
                    "|X X   X O X   |\n"
                    "|X O X X X O   |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    action1, _ = alphabeta(board, PLAYER2, np.int8(2))
    action2, _ = alphabeta(board, PLAYER1, np.int8(2))
    
    assert action1 == 2
    assert action2 == 2

def test_alphabeta_chooses_double_threat():
    from agents.agent_minimax.minimax import alphabeta

    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|O   X X     O |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    action, _ = alphabeta(board, PLAYER1, np.int8(4))
    
    assert action == 4

def test_alphabeta_chooses_double_threat_over_single():
    from agents.agent_minimax.minimax import alphabeta

    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|    O X       |\n"
                    "|O   X X     O |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    action, _ = alphabeta(board, PLAYER1, np.int8(2))
    
    assert action == 4

def test_alphabeta_chooses_to_fill_space():
    from agents.agent_minimax.minimax import alphabeta

    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|      x       |\n"
                    "|    O O       |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    action, _ = alphabeta(board, PLAYER2, np.int8(4))
    
    assert action == 4

def test_alphabeta_multi_win_situation():
    from agents.agent_minimax.minimax import alphabeta

    pretty_board = ("|==============|\n"
                    "|      X   X   |\n"
                    "|      X X O   |\n"
                    "|      X O O   |\n"
                    "|      O O O   |\n"
                    "|  O   X X X   |\n"
                    "|O O X X X O   |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    board = string_to_board(pretty_board)
    action1, _ = alphabeta(board, PLAYER1, np.int8(4))
    action2, _ = alphabeta(board, PLAYER2, np.int8(4))

    assert action1 == 2
    assert action2 == 2

    