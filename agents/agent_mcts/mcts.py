import numpy as np
from agents.common import *
import random
from typing import Optional, Tuple

ITERATIONS = 1000

class Node():
    
    def __init__(self, board, parent):
        self.board = board
        
        if is_terminal_board(self.board):
            self.is_terminal = True
        else:
            self.is_terminal = False
            
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.visits = 0
        self.total_score = 0
        self.children = {}
        
    def get_valid_columns(self):
        return np.where(self.board[-1, :] == NO_PLAYER)[0] 
    
    def search(self, initial_state):
        
        self.root = Node(initial_state, None)
        
        for iteration in range(ITERATIONS):
            # selection
            node = self.select(self.root)
            # simulation
            score = self.rollout(node.board)
            # backprop
            self.backpropogate(node, score)
            
        try:
            return self.get_best_move(self.root, 0)
        
        except:
            pass
        
    def select(self, node):
        
        
    def rollout():
        pass
    
    def backpropogate(node, score):
        pass
    
    def get_best_move(self, node, exploration):
        best_score = np.NINF
        best_moves = []
        
        for child_node in node.children.values():
            # TO DO: define current player
            
            move_score = child_node.score / child_node.visits + exploration * np.sqrt(np.log(node.visits/child_node.visits))
        
        if move_score > best_score:
            best_score = move_score
            best_moves = [child_node]
        elif move_score == best_score:
            best_moves.append(child_node)
            
        return np.random.choice(best_moves)
        

def mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[PlayerAction, Optional[SavedState]]:
  
    valid_columns = np.where(board[-1,:] == NO_PLAYER)[0]
    action = np.random.choice(valid_columns)

    return action, saved_state
        
def generate_move_mcts(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[SavedState],
    depth=np.int8(4)
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Runs the minimax algorithm and returns best action.
    """
    action, value = mcts(board, player, depth, True)
    return action, saved_state