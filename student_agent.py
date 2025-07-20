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
import pygame


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
    def __init__(self, window_size=(800,800)):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.window_size = window_size
        self.cell_size = window_size[0] // self.size

        pygame.font.init()
        self.font = pygame.font.SysFont('arial', self.cell_size // 4, bold=True)

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
    
    def pygame_render(self, surface, action=None):
        """
        Draw the current board state into the given Pygame surface.
        surface: a pygame.Surface of size self.window_size
        """
        # fill background
        surface.fill(pygame.Color("#bbada0"))

        for i in range(self.size):
            for j in range(self.size):
                val = self.board[i, j]
                # tile rectangle
                rect = pygame.Rect(
                    j * self.cell_size + 5,
                    i * self.cell_size + 5,
                    self.cell_size - 10,
                    self.cell_size - 10
                )
                pygame.draw.rect(surface, pygame.Color(COLOR_MAP.get(val, "#3c3a32")), rect, border_radius=5)

                if val != 0:
                    txt_surf = self.font.render(str(val), True,
                                                pygame.Color(TEXT_COLOR.get(val, "#f9f6f2")))
                    txt_rect = txt_surf.get_rect(center=rect.center)
                    surface.blit(txt_surf, txt_rect)

        # Optional status bar at bottom
        info = f"Score: {self.score}"
        if action is not None:
            info += f"  Move: {self.actions[action]}"
        info_surf = self.font.render(info, True, pygame.Color("#776e65"))
        surface.blit(info_surf, (10, self.window_size[1] - self.cell_size // 2))

        # finally flip/update
        pygame.display.flip()

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

    def __getstate__(self):
        """Customize what gets pickled for deepcopy"""
        state = self.__dict__.copy()
        # Remove non-pickleable objects
        if 'font' in state:
            del state['font']
        return state

    def __setstate__(self, state):
        """Restore state after unpickling"""
        self.__dict__.update(state)
        # Recreate the font after unpickling if window size is available
        if hasattr(self, 'window_size') and hasattr(self, 'size'):
            cell_size = self.window_size[0] // self.size
            self.font = pygame.font.SysFont('arial', cell_size // 4, bold=True)


########################################################################
#                       given environment above                        #
########################################################################

class Pattern:
    def __init__(self, pattern: list, iso=8):
        self.pattern = pattern
        self.iso = iso
        self.weights = None
        self.isom = self._create_isomorphic_patterns()

    def _create_isomorphic_patterns(self) -> list:
        isom = []
        for i in range(self.iso):
            idx = self._rotate_mirror_pattern([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], i)
            patt = [idx[p] for p in self.pattern]
            isom.append(patt)
        return isom

    def _rotate_mirror_pattern(self, base: list, rot: int) -> list:
        board = np.array(base, dtype=int).reshape(4,4)
        if rot >= 4:
            board = np.fliplr(board)
        board = np.rot90(board, rot % 4)
        return board.flatten().tolist()

    def load_weights(self, weights: list):
        self.weights = weights

    def estimate(self, board: np.ndarray) -> float:
        total = 0.0
        for iso in self.isom:
            index = self._get_index(iso, board)
            total += self.weights[index]
        return total

    def _get_index(self, pattern: list, board: np.ndarray) -> int:
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
    def __init__(self, bin_path: str):
        self.patterns = []
        self._load_binary(bin_path)

    def _load_binary(self, path: str):
        with open(path, 'rb') as f:
            num_features = struct.unpack('Q', f.read(8))[0]
            
            for _ in range(num_features):
                # Read feature name
                name_len = struct.unpack('I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                
                # Parse pattern from name (e.g., "4-tuple pattern 0123")
                pattern = [int(c, 16) for c in name.split()[-1]]
                
                # Create pattern and load weights
                p = Pattern(pattern)
                size = struct.unpack('Q', f.read(8))[0]
                weights = struct.unpack(f'{size}f', f.read(4*size))
                p.load_weights(weights)
                self.patterns.append(p)

    def estimate(self, board: np.ndarray) -> float:
        return sum(p.estimate(board) for p in self.patterns)


class MCTSNode:
    """Node for Monte Carlo Tree Search"""
    def __init__(self, action=None, parent=None):
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
    
    @property
    def ucb_score(self) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + 1.0 * math.sqrt(math.log(self.parent.visits) / self.visits)


class MCTS:
    """Monte Carlo Tree Search with depth 2 expetimax evaluation"""
    def __init__(self, model: CPPModel, exploration_weight=1.0):
        self.model = model
        self.exploration_weight = exploration_weight
    
    def search(self, env: Game2048Env, iterations=100, depth=2) -> int:
        """Perform MCTS search from current game state"""
        root = MCTSNode()
        legal_actions = [a for a in range(4) if env.is_move_legal(a)]

        for action in legal_actions:
            root.children.append(MCTSNode(action=action, parent=root))
        
        for _ in range(iterations):
            node = self._select(root)
            if depth == 1:
                value = self._simulate_depth1(copy.deepcopy(env), node)
            else:
                value = self._simulate_depth2(copy.deepcopy(env), node)
            self._backpropagate(node, value)
        
        return max(root.children, key=lambda c: c.value).action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: Choose child nodes using UCB until leaf node found"""
        while node.children:
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                return random.choice(unvisited)
            node = max(node.children, key=lambda c: c.ucb_score)
        return node

    def _simulate_depth2(self, env: Game2048Env, node: MCTSNode) -> float:
        """Simulation phase: Perform depth 2 expectimax evaluation"""
        original_score = env.score
        moved = self._execute_action(env, node.action)

        if not moved:
            return -float('inf')

        empty_cells = list(zip(*np.where(env.board == 0)))
        expected_value = 0.0

        if not empty_cells:
            return self._evaluate_terminal(env, original_score)
        
        for (x, y) in empty_cells:
            cell_prob = 1.0 / len(empty_cells)
            for tile_value, prob in [(2, 0.9), (4, 0.1)]:
                temp_env = copy.deepcopy(env)
                temp_env.board[x, y] = tile_value

                if temp_env.is_game_over():
                    contribution = -50000
                else:
                    contribution = self.model.estimate(temp_env.board) + (temp_env.score - original_score)
                
                expected_value += cell_prob * prob * contribution
        
        return expected_value

    def _simulate_depth1(self, env: Game2048Env, node: MCTSNode) -> float:
        """Simulation for depth 1: Only evaluate immediate move"""
        original_score = env.score
        moved = self._execute_action(env, node.action)

        if not moved:
            return -float('inf')
        
        return self.model.estimate(env.board) + (env.score - original_score)

    def _evaluate_terminal(self, env: Game2048Env, original_score: float) -> float:
        """Handle terminal state evaluation"""
        if env.is_game_over():
            return -50000
        return self.model.estimate(env.board) + (env.score - original_score)

    def _execute_action(self, env: Game2048Env, action: int) -> bool:
        """Execute an action on the environment"""
        action_map = {
            0: env.move_up,
            1: env.move_down,
            2: env.move_left,
            3: env.move_right
        }
        return action_map[action]()

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagation phase: Update node statistics"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent


# def board_to_cpp_format(py_board):
#     """Convert Python board (2D array) to C++ style 64-bit integer"""
#     cpp_board = 0
#     for i in range(4):
#         row = 0
#         for j in range(4):
#             tile = py_board[i][j]
#             val = 0 if tile == 0 else int(np.log2(tile))
#             row |= (val & 0xF) << (j * 4)
#         cpp_board |= row << (i * 16)
#     return cpp_board

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
    
    mcts = MCTS(model=model)
    best_action = mcts.search(env, iterations=4, depth=1)
    return best_action


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