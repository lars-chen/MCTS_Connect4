disable_jit = True
if disable_jit:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
from numba import njit
from agents.common import PLAYER1, PLAYER2, NO_PLAYER, GameState, BoardPiece, SavedState, PlayerAction, check_end_state, is_terminal_board, apply_player_action
from typing import Optional, Tuple


def get_valid_actions(board):
    return np.where(board[-1, :] == NO_PLAYER)[0]


def other_player(player):
    return PLAYER1 if player is PLAYER2 else PLAYER2


class Node():
    def __init__(self, board, parent, player):

        # State details
        self.board = board
        self.player = player
        self.parent = parent
        self.children = {}
        self.unplayed_actions = get_valid_actions(board)
        self.is_terminal = True if is_terminal_board(
            board, other_player(player)) else False

        # Monte Carlo Metrics
        self.visits = 0
        self.wins = 0


# Monte Carlo Tree Search
class MCTS(object):
    def __init__(self,
                 iterations,
                 current_player,
                 current_board,
                 exploration_const=1 / np.sqrt(2)):

        self.iterations = iterations
        self.exploration_const = exploration_const
        self.current_player = current_player
        self.rootnode = Node(current_board, None, current_player)

    @njit()
    def get_best_action(self):
        for _ in range(self.iterations):
            node = self.select(self.rootnode)
            result = self.simulate(node)
            self.backpropogate(node, result)

        most_visits = -1
        for action in self.rootnode.children:
            if self.rootnode.children[action].visits > most_visits:
                best_action = action
        return best_action

    @njit()
    def select(self, node):
        while not node.is_terminal:
            if len(node.unplayed_actions) == 0:
                node = self.get_best_child(node)
            else:
                node = self.expand(node)
        return node

    def expand(self, node):
        player = node.player
        child_player = other_player(node.player)
        action = np.random.choice(node.unplayed_actions)
        child_board = apply_player_action(node.board, action, player)
        node.unplayed_actions = node.unplayed_actions[
            node.unplayed_actions != action]
        child_node = Node(child_board, node, child_player)
        node.children[action] = child_node
        return child_node

    def simulate(self, node):
        board = node.board
        player = node.player
        while len(get_valid_actions(board)) != 0 and not is_terminal_board(
                board, other_player(player)):
            action = np.random.choice(get_valid_actions(board))
            board = apply_player_action(board, action, player)
            player = other_player(player)
        result = 1 if check_end_state(
            board, self.current_player) is GameState.IS_WIN else 0
        return result

    @njit()
    def backpropogate(self, node, result):
        while node is not None:
            node.visits += 1
            if other_player(node.player) == self.current_player:
                node.wins += result
            node = node.parent
            
    @njit()
    def get_best_child(self, node):
        best_score = np.NINF
        for child in node.children.values():
            move_score = child.wins / child.visits + self.exploration_const * np.sqrt(
                np.log(child.visits / child.visits))
            if move_score > best_score:
                best_score = move_score
                best_child = child

        return best_child

def generate_move_mcts(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[SavedState],
    iterations=1000
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Runs the mcts algorithm and returns best action.
    """
    mcts_search = MCTS(iterations, player, board)
    action = mcts_search.get_best_action()
    return action, saved_state
