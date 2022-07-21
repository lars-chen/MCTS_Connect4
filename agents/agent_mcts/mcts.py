disable_jit = True
if disable_jit:
    import os

    os.environ["NUMBA_DISABLE_JIT"] = "1"

from agents.common import (
    PLAYER1,
    PLAYER2,
    NO_PLAYER,
    GameState,
    BoardPiece,
    SavedState,
    PlayerAction,
    check_end_state,
    is_terminal_board,
    apply_player_action,
)
import numpy as np
import time
from numba import njit
from typing import Optional, Tuple


def get_valid_actions(board):
    """Returns 1D array of valid actions that can be played into.

    Args:
        board (ndarray): Board to evaluate which moves are valid.

    Returns:
        Array: 1D array of valid actions.
    """
    return np.where(board[-1, :] == NO_PLAYER)[0]


def other_player(player):
    """Returns the opposing player
    """
    return PLAYER1 if player is PLAYER2 else PLAYER2


class Node:
    def __init__(self, board, parent, player):
        """Class constructor for a node in the tree. Attributes are included to describe
        the game state of the node and statistics for monte carlo tree search.

        Args:
            board (Numpy.ndarray): Array describing the board state of the node.
            parent (Node): Parent node of current node.
            player (BoardPiece): Current player to play into the state of the node.
        """
        # State details
        self.board = board
        self.player = player
        self.parent = parent
        self.children = {}
        self.unplayed_actions = get_valid_actions(board)
        if is_terminal_board(board, other_player(player)) or is_terminal_board(
            board, player
        ):
            self.is_terminal = True
        else:
            self.is_terminal = False

        # Monte Carlo Metrics
        self.visits = 0
        self.wins = 0

# Monte Carlo Tree Search
class MCTS(object):
    def __init__(
        self,
        current_player,
        current_board,
        iterations=False,
        timeout=4,
        exploration_const=1 / np.sqrt(2),
    ):
        """Class constructor for a Monte Carlo tree search

        Args:
            current_player (BoardPiece): _description_
            current_board (numpy.ndarray): _description_
            iterations (int, optional): _description_. Defaults to 2000.
            timeout (int, optional): _description_. Defaults to 5.
            exploration_const (int, optional): _description_. Defaults to 1/np.sqrt(2).
        """
        self.timeout = timeout
        self.iterations = iterations
        self.exploration_const = exploration_const
        self.current_player = current_player
        self.rootnode = Node(current_board, None, current_player)

    def search(self, node):
        """Run one iteration of monte carlo tree search on given node.

        Args:
            node (Node): 
        """
        node = self.select()
        result = self.simulate(node)
        self.backpropogate(node, result)
        
    @njit()
    def get_best_action(self):
        """Runs iterations of the monte carlo tree search algorithm as defined by the MCTS class
        to build a tree search with win and visits stored in each node. After the iterative tree
        search, returns the child node of the root with the most visits.
        

        Returns:
            Int: An action from 0 to 6 to play on connect 4 board.
        """
        if self.timeout:
            timeout = time.time() + self.timeout
            while time.time() < timeout:
                self.search(self.rootnode)
        elif self.iterations:
            for _ in range(self.iterations):
                self.search(self.rootnode)
        
        most_visits = -1
        for action in self.rootnode.children:
            child_visits = self.rootnode.children[action].visits
            if child_visits > most_visits:
                best_action = action
                most_visits = child_visits
        return best_action

    @njit()
    def select(self):
        """Successively selects the best child node of fully expanded nodes based off UCT formula 
        until a terminal node is reached. If a non-fully expanded node is encountered, select function
        will add a random child node from a valid action that is unplayed.

        Returns:
            Node: returns the node that was expanded or the best terminal child node.
        """
        node = self.rootnode
        while not node.is_terminal:
            if len(node.unplayed_actions) == 0:
                node = self.get_best_child(node)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        """Adds a random child node to an input node that has unplayed actions.

        Args:
            node (Node): Input a node with valid unplayed actions. 

        Returns:
            Node: Returns the child node that was added.
        """
        action = np.random.choice(node.unplayed_actions)
        child_board = apply_player_action(node.board, action, node.player)
        node.unplayed_actions = node.unplayed_actions[node.unplayed_actions != action]  # remove played action from node.unplayed_actions
        child_node = Node(child_board, node, other_player(node.player))
        node.children[action] = child_node  # expand child into node
        return child_node

    @njit()
    def simulate(self, node):
        """ Runs a random rollout to a terminal state of the board.

        Args:
            node (Node): Takes in a node to simulate a random game off of.

        Returns:
            Int: 1 if game was a win for the root node player, otherwise 0.
        """
        board = node.board
        player = node.player
        while len(get_valid_actions(board)) != 0 and not is_terminal_board(
            board, other_player(player)
        ):
            action = np.random.choice(get_valid_actions(board))
            board = apply_player_action(board, action, player)
            player = other_player(player)
        result = (
            1 if check_end_state(board, self.current_player) is GameState.IS_WIN else 0
        )
        return result

    @njit()
    def backpropogate(self, node, result):
        """Update tree statistics

        Args:
            node (Node): node to backpropogate from
            result (Int): result of random rollout
        """
        while node is not None:
            node.visits += 1
            if other_player(node.player) == self.current_player:
                node.wins += result
            node = node.parent

    @njit()
    def get_best_child(self, node):
        """Returns best child of given node as per the UCT formula.

        Args:
            node (Node): Takes a Node to search through its child nodes.

        Returns:
            Node: best child node of the input node.
        """
        best_score = np.NINF
        for child in node.children.values():
            move_score = child.wins / child.visits + self.exploration_const * np.sqrt(
                np.log(node.visits / child.visits)
            )
            if move_score > best_score:
                best_score = move_score
                best_child = child

        return best_child


def generate_move_mcts(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[SavedState],
    timeout=4,
    iterations=False,
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """Returns best action as defined by MCTS parameters.

    Args:
        board (np.ndarray): current state of the board to be played into.
        player (BoardPiece): current player to place a piece
        saved_state (Optional[SavedState]): not used
        iterations (int, optional): Number of MCTS iterations. Defaults to 1000. 
        timeout (float, optional): Seconds until search times out. Defaults to 4.

    Returns:
        Tuple[PlayerAction, Optional[SavedState]]: A tuple of the best action as per MCTS and saved state which is not used currently.
    """
    mcts_search = MCTS(player, board, iterations, timeout)
    action = mcts_search.get_best_action()

    return action, saved_state
