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
    
    
def test_select_node():
    board = initialize_game_state()
    mcts = MCTS(PLAYER1, board)
    child = mcts.expand(mcts.rootnode)
    
    assert mcts.select() == child
    

def test_get_best_action():
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
    mcts = MCTS(PLAYER2, board, 1, 5)
    mcts.get_best_action() == 6
