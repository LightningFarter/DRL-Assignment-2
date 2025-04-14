# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import struct

from collections import defaultdict


COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None, savepath=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        if savepath:
            fig.savefig(savepath)
            plt.close()
        else:
            plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


########################################################################
#                       given environment above                        #
########################################################################

class Pattern:
    def __init__(self, pattern, iso=8):
        self.pattern = pattern
        self.iso = iso
        self.weights = None
        self.isom = self._create_isomorphic_patterns()

    def _create_isomorphic_patterns(self):
        isom = []
        for i in range(self.iso):
            idx = self._rotate_mirror_pattern([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], i)
            patt = [idx[p] for p in self.pattern]
            isom.append(patt)
        return isom

    def _rotate_mirror_pattern(self, base, rot):
        board = np.array(base, dtype=int).reshape(4,4)
        if rot >= 4:
            board = np.fliplr(board)
        board = np.rot90(board, rot % 4)
        return board.flatten().tolist()

    def load_weights(self, weights):
        self.weights = weights

    def estimate(self, board):
        total = 0.0
        for iso in self.isom:
            index = self._get_index(iso, board)
            total += self.weights[index]
        return total

    def _get_index(self, pattern, board):
        index = 0
        for i, pos in enumerate(pattern):
            tile = board[pos//4][pos%4]
            if tile == 0: 
                val = 0
            else:
                val = int(np.log2(tile))
            index |= (val & 0xF) << (4 * i)
        return index


class CPPModel:
    def __init__(self, bin_path):
        self.patterns = []
        self._load_binary(bin_path)

    def _load_binary(self, path):
        with open(path, 'rb') as f:
            num_features = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_features):
                # Read feature name
                name_len = struct.unpack('I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                
                # Parse pattern from name (e.g., "4-tuple pattern 0123")
                print(name)
                pattern = [int(c, 16) for c in name.split()[-1]]
                
                # Create pattern and load weights
                p = Pattern(pattern)
                size = struct.unpack('Q', f.read(8))[0]
                weights = struct.unpack(f'{size}f', f.read(4*size))
                p.load_weights(weights)
                self.patterns.append(p)

    def estimate(self, board):
        return sum(p.estimate(board) for p in self.patterns)


class Node:
    def __init__(self, action=None, parent=None):
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

def select_child(node: Node) -> Node:
    if not node.children:
        return None
    
    unvisited = [c for c in node.children if c.visits == 0]
    if unvisited:
        return random.choice(unvisited)

    C = 1.0
    best_score = -float('inf')
    best_child = None
    for child in node.children:
        if child.visits > 0:
            ucb = (child.value / child.visits) + C * math.sqrt(math.log(node.visits) / child.visits)
            if ucb > best_score:
                best_score = ucb
                best_child = child
    return best_child


def board_to_cpp_format(py_board):
    """Convert Python board (2D array) to C++ style 64-bit integer"""
    cpp_board = 0
    for i in range(4):
        row = 0
        for j in range(4):
            tile = py_board[i][j]
            val = 0 if tile == 0 else int(np.log2(tile))
            row |= (val & 0xF) << (j * 4)
        cpp_board |= row << (i * 16)
    return cpp_board

# Initialize model (load once at startup)
model = CPPModel('2048.bin')
# for p in model.patterns:
#     print(p.weights[0:5], end='\n\n')

# ts = np.array([
#     [2, 4, 0, 0],
#     [8, 16, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 0]
# ])
# print(hex(board_to_cpp_format(ts)))
# print(model.estimate(ts))



def get_action(state, score):
    env = Game2048Env()
    env.board = np.array(state)
    env.score = score
    
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    
    if not legal_moves:
        return 0  # Should never happen
    
    root = Node() # MCTS root
    
    for action in legal_moves:
        child = Node(action=action, parent=root)
        root.children.append(child)
    
    num_iterations = len(legal_moves)
    for _ in range(num_iterations):
        child = select_child(root)

        temp_env = copy.deepcopy(env)

        # shallow rollout
        if child.action == 0:
            temp_env.move_up()
        elif child.action == 1:
            temp_env.move_down()
        elif child.action == 2:
            temp_env.move_left()
        elif child.action == 3:
            temp_env.move_right()
        
        # heuristic with n-tuple estimator
        estimated_value = model.estimate(temp_env.board) + temp_env.score - score

        # simulate tiles filling
        empty_cells = list(zip(*np.where(temp_env.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            tile_value = 2 if random.random() < 0.9 else 4
            temp_env.board[x, y] = tile_value

        if temp_env.is_game_over():
            estimated_value = -50000

        child.value += estimated_value
        child.visits += 1
        root.visits += 1
    
    best_child = max(root.children, key=lambda c: c.value)
    return best_child.action


### simple TD trail for testing
# def get_action(state, score):
#     env = Game2048Env()
#     env.board = np.array(state)
#     env.score = score
    
#     legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    
#     if not legal_moves:
#         return 0  # Should never happen
    
#     best_value = -float('inf')
#     best_action = legal_moves[0]
    
#     for action in legal_moves:
#         # Simulate move
#         temp_env = copy.deepcopy(env)

#         if action == 0:
#             temp_env.move_up()
#         elif action == 1:
#             temp_env.move_down()
#         elif action == 2:
#             temp_env.move_left()
#         elif action == 3:
#             temp_env.move_right()

#         reward = temp_env.score
#         done = temp_env.is_game_over()

#         if done:
#             continue
        
#         # Get afterstate value
#         afterstate = temp_env.board
#         reward -= score
#         value = model.estimate(afterstate) + reward
        
#         if value > best_value:
#             best_value = value
#             best_action = action
            
#     return best_action