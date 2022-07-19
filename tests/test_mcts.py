from mimetypes import init
from agents.common import *
from agents.agent_mcts.mcts import *


def test_other_player():  
    assert other_player(PLAYER1) is PLAYER2
    assert other_player(PLAYER2) is PLAYER1
    
    
def 