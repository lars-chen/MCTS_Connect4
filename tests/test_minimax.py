import numpy as np
from agents.common import *

def test_valid_columns_on_on_intitial_state():
    from agents.agent_minimax.minimax import sorted_valid_columns
    
    board = initialize_game_state()
    columns = sorted_valid_columns(board)
    assert len(columns) == 7


def test_alphabeta_on_initial_state():
    from agents.agent_minimax.minimax import alphabeta

    board = initialize_game_state
    value = alphabeta(board, PLAYER1, 4, True)
    assert type(value) == int
