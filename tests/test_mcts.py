from mimetypes import init
from agents.common import *
from agents.agent_mcts.mcts import *


def test_other_player():
    from agents.agent_mcts.mcts import other_player
    assert other_player(PLAYER1) is PLAYER2
    assert other_player(PLAYER2) is PLAYER1


def test_get_valid_actions():
    pretty_board = (
        "|==============|\n"
        "|    X         |\n"
        "|    X         |\n"
        "|    X         |\n"
        "|    X         |\n"
        "|    X         |\n"
        "|    X   X X   |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |"
    )
    board = string_to_board(pretty_board)
    assert (get_valid_actions(board) == np.array([0, 1, 3, 4, 5, 6])).all()


def test_expand_node():
    board = initialize_game_state()
    mcts = MCTS(PLAYER1, board, iterations=1, timeout=5)
    node = mcts.expand(mcts.rootnode)
    assert len(mcts.rootnode.children) == 1
    

def test_get_best_action_P1():
    pretty_board = (
        "|==============|\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|              |\n"
        "|      O       |\n"
        "|    O X X X   |\n"
        "|==============|\n"
        "|0 1 2 3 4 5 6 |\n"
    )
    board = string_to_board(pretty_board)
    mcts = MCTS(PLAYER2, board, 1000, False)
    assert mcts.get_best_action() == 6


def test_get_best_action_P2():
    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|    X         |\n"
                    "|    X O       |\n"
                    "|  X O O O     |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    
    board = string_to_board(pretty_board)
    mcts = MCTS(PLAYER2, board, 2000, False)
    assert mcts.get_best_action() == 5


def test_backpropogate_visits():
    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|    X O       |\n"
                    "|  X O O O     |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    
    board = string_to_board(pretty_board)
    mcts = MCTS(PLAYER2, board, 7, False)
    _ = mcts.get_best_action()
    for i in range(7):
        assert mcts.rootnode.children[i].visits == 1
    
        
def test_backpropogate_wins():
    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|    X O       |\n"
                    "|  X X O O   O |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    
    board = string_to_board(pretty_board)
    mcts = MCTS(PLAYER2, board, 7, False)
    _ = mcts.get_best_action()
    assert mcts.rootnode.children[5].wins == 1
    
    
def test_simulate_on_winning_node():
    pretty_board = ("|==============|\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|              |\n"
                    "|    X O       |\n"
                    "|  X X O O   O |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    
    board = string_to_board(pretty_board)
    mcts = MCTS(PLAYER2, board, 7, False)
    _ = mcts.get_best_action()
    result = mcts.simulate(mcts.rootnode.children[5])
    assert result == 1
    
def test_simulate_on_future_win():
    pretty_board = ("|==============|\n"
                    "|X X O X     X |\n"
                    "|O O X O     O |\n"
                    "|X X O X     X |\n"
                    "|O O X O     O |\n"
                    "|X O X O X   X |\n"
                    "|O X X O O   O |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    
    board = string_to_board(pretty_board)
    mcts = MCTS(PLAYER2, board, 7, False)
    _ = mcts.get_best_action()
    result = mcts.simulate(mcts.rootnode.children[5])
    assert result == 1
    
def test_expand_on_limited_actions():
    pretty_board = ("|==============|\n"
                    "|X X O X     X |\n"
                    "|O O X O     O |\n"
                    "|X X O X     X |\n"
                    "|O O X O     O |\n"
                    "|X O X O X   X |\n"
                    "|O X X O O   O |\n"
                    "|==============|\n"
                    "|0 1 2 3 4 5 6 |")
    
    board = string_to_board(pretty_board)
    mcts = MCTS(PLAYER2, board, 10, False)
    _ = mcts.get_best_action()
    assert len(mcts.rootnode.children) == 2
