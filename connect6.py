from typing import Optional, Tuple, List, Dict, Any
import sys
import numpy as np
import random
from collections import defaultdict

def lines_through_cell(board: np.ndarray, cell:Tuple[int, int]) -> List[List[Tuple[int, int]]]:
    """
    Returns all lines of exactly six positions passing through the given cell (i, j)
    in the four directions: horizontal, vertical, main diagonal, anti-diagonal.
    Each line is a list of (row, col) tuples, constrained within the board.
    """
    n = board.shape[0]
    i, j = cell
    lines = []

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for di, dj in directions:
        line = []
        for k in range(6):
            r, c = i + k * di, j + k * dj
            if not (0 <= r < n and 0 <= c < n):
                break
            line.append((r, c))
        for k in range(1, 6):
            r, c = i - k * di, j - k * dj
            if not (0 <= r < n and 0 <= c < n):
                break
            line.append((r, c))
        line.sort()
        for start_idx in range(max(0, len(line) - 5)):
            six_line = line[start_idx:start_idx + 6]
            if len(six_line) == 6 and (i, j) in six_line:
                lines.append(six_line)
    
    return lines

def is_still_dangerous(board: np.ndarray, cell: Tuple[int, int], opponent: int) -> bool:
    """
    Checks if the cell is still part of a threat for the opponent.
    A cell is dangerous if it’s in a line with 5 opponent stones and 1 empty,
    or 4 opponent stones and 2 empty.
    """
    for line in lines_through_cell(board, cell):
        stones = [board[i, j] for i, j in line]
        if all(s in [opponent, 0] for s in stones):
            opp_count = sum(s == opponent for s in stones)
            empty_count = sum(s == 0 for s in stones)
            if (opp_count == 5 and empty_count >= 1) or (opp_count == 4 and empty_count >= 2):
                return True
    return False

def evaluate_board(board: np.ndarray) -> float:
    """
    Evaluates the board from Black's (color 1) perspective.
    Returns 1.0 if Black wins, -1.0 if White wins, else a heuristic score.
    """
    n = board.shape[0]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    max_line_black = 0
    max_line_white = 0
    for i in range(n):
        for j in range(n):
            if board[i, j] == 1:  # Black
                for di, dj in directions:
                    count = 1
                    r, c = i + di, j + dj
                    while 0 <= r < n and 0 <= c < n and board[r, c] == 1:
                        count += 1
                        r += di
                        c += dj
                    max_line_black = max(max_line_black, count)
            elif board[i, j] == 2:  # White
                for di, dj in directions:
                    count = 1
                    r, c = i + di, j + dj
                    while 0 <= r < n and 0 <= c < n and board[r, c] == 2:
                        count += 1
                        r += di
                        c += dj
                    max_line_white = max(max_line_white, count)
    if max_line_black >= 6:
        return 1.0
    if max_line_white >= 6:
        return -1.0
    return (max_line_black / 6.0) ** 3 - (max_line_white / 6.0) ** 2

def count_opponent_neighbors(board: np.ndarray, cell: Tuple[int, int], color: int) -> int:
    """
    Counts the number of opponent stones adjacent to the given cell (i, j).
    
    Args:
        board: NumPy array representing the game board (0: empty, 1: Black, 2: White).
        cell: Tuple of (row, col) coordinates for the cell.
        color: Current player's color (1 for Black, 2 for White).
    
    Returns:
        Number of neighboring cells occupied by the opponent's stones.
    """
    i, j = cell
    n = board.shape[0]
    opponent = 3 - color
    
    # Define the eight neighboring directions: (up, down, left, right, diagonals)
    directions = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
        (-1, -1), # up-left
        (-1, 1),  # up-right
        (1, -1),  # down-left
        (1, 1)    # down-right
    ]
    
    count = 0
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < n and 0 <= nj < n:  # Check if neighbor is within board
            if board[ni, nj] == opponent:
                count += 1
    
    return count

def recommended_moves(board: np.ndarray, color: int, size=10) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Generates recommended move pairs for the given board and color."""
    n = board.shape[0]
    opponent = 3 - color
    # Check instant win
    for i in range(n):
        for j in range(n):
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                empty_cells = []
                for k in range(6, 0, -1):
                    ni = i + di * k
                    nj = j + dj * k
                    if not (0 <= ni < n and 0 <= nj < n):
                        break
                    if board[ni, nj] == 3 - color:
                        break
                    if board[ni, nj] == 0:
                        empty_cells.append((ni, nj))
                else:
                    if len(empty_cells) == 1:
                        for _i in range(n):
                            for _j in range(n):
                                if board[_i, _j] == 0 and (_i, _j) != empty_cells[0]:
                                    return [(empty_cells[0], (_i, _j))]
                    if len(empty_cells) == 2:
                        return [tuple(empty_cells)]
    # Check for potential threats
    visited = set()
    dangerous_cells = []
    for i in range(n):
        for j in range(n):
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                empty_cells = []
                for k in range(6, 0, -1):
                    ni = i + di * k
                    nj = j + dj * k
                    if not (0 <= ni < n and 0 <= nj < n):
                        break
                    if board[ni, nj] == color:
                        break
                    if board[ni, nj] == 0:
                        empty_cells.append((ni, nj))
                else:
                    if len(empty_cells) == 1:
                        if empty_cells[0] not in visited:
                            dangerous_cells.append(empty_cells[0])
                            visited.add(empty_cells[0])
                    if len(empty_cells) == 2 and (abs(empty_cells[0][0] - empty_cells[1][0]) == 5 or abs(empty_cells[0][1] - empty_cells[1][1]) == 5):
                        return [tuple(empty_cells)]
    
    dangerous_cells = list(set(dangerous_cells))

    if len(dangerous_cells) >= 2:
        return [(dangerous_cells[0], dangerous_cells[1])]
                
    for i in range(n):
        for j in range(n):
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                empty_cells = []
                for k in range(6, 0, -1):
                    ni = i + di * k
                    nj = j + dj * k
                    if not (0 <= ni < n and 0 <= nj < n):
                        break
                    if board[ni, nj] == color:
                        break
                    if board[ni, nj] == 0:
                        empty_cells.append((ni, nj))
                else:
                    if len(empty_cells) == 1:
                        if empty_cells[0] not in visited:
                            dangerous_cells.append(empty_cells[0])
                            visited.add(empty_cells[0])
                    if len(empty_cells) == 2:
                        return [tuple(empty_cells)]
                
                if len(dangerous_cells) == 2:
                    return [tuple(dangerous_cells)]
    
    if len(dangerous_cells) == 1:
        board[dangerous_cells[0]] = color
        left = 1
    else:
        left = 2

    # for i in range(n):
    #     for j in range(n):
    #         for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
    #             empty_cells = []
    #             for k in range(6, 0, -1):
    #                 ni = i + di * k
    #                 nj = j + dj * k
    #                 if not (0 <= ni < n and 0 <= nj < n):
    #                     break
    #                 if board[ni, nj] == color:
    #                     break
    #                 if board[ni, nj] == 0:
    #                     empty_cells.append((ni, nj))
    #             else:
    #                 if len(empty_cells) == 1:
    #                     if empty_cells[0] not in visited:
    #                         dangerous_cells.append(empty_cells[0])
    #                         visited.add(empty_cells[0])
    #                 if len(empty_cells) == 2:
    #                     return [tuple(empty_cells)]
                
    #             if len(dangerous_cells) == 2:
    #                 return [tuple(dangerous_cells)]

    # if len(dangerous_cells) == 1:
    #     board[dangerous_cells[0]] = color
    #     left = 1
    # else:
    #     left = 2

    # Find possible pairs close to each other
    candidates = defaultdict(float)  # ((ni, nj), (ni2, nj2)) -> score
    for i in range(n):
        for j in range(n):
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                color_count = [0, 0, 0]
                empty_cells = []
                for k in range(6, 0, -1):
                    ni = i + di * k
                    nj = j + dj * k
                    if not (0 <= ni < n and 0 <= nj < n):
                        break
                    color_count[board[ni, nj]] += 1
                    if color_count[1] > 0 and color_count[2] > 0:
                        break
                    if board[ni, nj] == 0:
                        empty_cells.append((ni, nj))
                else:
                    if left == 1:
                        for x in empty_cells:
                            candidates[(x, dangerous_cells[0])] += 2 * color_count[color] ** 2 + color_count[3 - color] ** 2
                    else:
                        for x in empty_cells:
                            for y in empty_cells:
                                if x != y:
                                    candidates[(x, y)] += 2 * color_count[color] ** 2 + color_count[3 - color] ** 2
    
    best_moves = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:size * 2]
    best_moves = [(x[0][0], x[0][1]) for i, x in enumerate(best_moves) if i % 2 == 0]
    return best_moves

class MCTSNode:
    def __init__(self, board: np.ndarray, color, parent=None):
        self.board = board.copy()
        self.color = color
        self.parent: Optional[MCTSNode] = parent
        self.children: Dict[Tuple[Tuple[int, int], Tuple[int, int]], MCTSNode] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = recommended_moves(board, color)

    def fully_expanded(self):
        return len(self.untried_actions) == 0

class MCTS:
    def __init__(self, iterations=300, exploration_constant=1.41):
        self.iterations = iterations
        self.c = exploration_constant

    def select_child(self, node: MCTSNode) -> MCTSNode:
        def value(child: MCTSNode):
            assert child.visits > 0, "Child node has no visits"
            return child.total_reward / child.visits + self.c * np.sqrt(np.log(node.visits) / child.visits)
        
        return max(node.children.values(), key=value)

    def expand(self, node: MCTSNode) -> MCTSNode:
        action = node.untried_actions.pop()
        board = node.board.copy()
        color = node.color
        board[action[0]] = color
        board[action[1]] = color
        child_node = MCTSNode(board, 3 - color, parent=node)
        node.children[action] = child_node
        return child_node
    
    def alphabeta(self, board: np.ndarray, color: int, depth: int, alpha: float, beta: float) -> float:
        """Alpha-beta search to evaluate positions, returning score from Black's perspective."""
        if depth == 0:
            return evaluate_board(board)
        moves = recommended_moves(board, color, size=10)
        if len(moves) == 0:
            return evaluate_board(board)
        if color == 1:  # Black, maximizing
            value = -np.inf
            for move in moves:
                new_board = board.copy()
                new_board[move[0]] = color
                new_board[move[1]] = color
                value = max(value, self.alphabeta(new_board, 3 - color, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:  # White, minimizing
            value = np.inf
            for move in moves:
                new_board = board.copy()
                new_board[move[0]] = color
                new_board[move[1]] = color
                value = min(value, self.alphabeta(new_board, 3 - color, depth - 1, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def rollout(self, board: np.ndarray, color: int) -> float:
        """Perform a shallow alpha-beta search for rollout, adjusting score to current player's perspective."""
        score = self.alphabeta(board, color, depth=0, alpha=-np.inf, beta=np.inf)
        return score if color == 1 else -score
    
    def backpropagate(self, node: MCTSNode, total_reward: float):
        while node is not None:
            node.visits += 1
            node.total_reward += total_reward
            total_reward = -total_reward  # Invert for opponent
            node = node.parent

    def terminate(self, board: np.ndarray) -> bool:
        """Check for Connect-6 win."""
        n = board.shape[0]
        for i in range(n):
            for j in range(n):
                if board[i, j] != 0:
                    color = board[i, j]
                    for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1)]:
                        count = 0
                        for k in range(6):
                            ni = i + di * k
                            nj = j + dj * k
                            if not (0 <= ni < n and 0 <= nj < n):
                                break
                            if board[ni, nj] == color:
                                count += 1
                            else:
                                break
                        if count >= 6:
                            return True
        return False

    def simulate(self, root: MCTSNode):
        node = root
        while not self.terminate(node.board) and node.fully_expanded():
            node = self.select_child(node)
            
        if not node.fully_expanded():
            node = self.expand(node)

        reward = self.rollout(node.board, node.color)
        self.backpropagate(node, reward)



class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return
        
        ###################################################
        #                  implement here                 #
        ###################################################
        if self.board.sum() == 0:
            move_str = "K10"
            self.play_move(color, move_str)
            print(f"{move_str}\n\n", end='', flush=True)
            print(move_str, file=sys.stderr)
            return
        
        root = MCTSNode(self.board, self.turn)
        mcts = MCTS(iterations=50)
        for _ in range(mcts.iterations):
            mcts.simulate(root)

        best_move = max(root.children.items(), key=lambda x: x[1].visits)
        selected = best_move[0]
        move_str = ','.join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)

        
        # if best_action is None:
        #     print("? No valid move")
        #     return

        self.play_move(color, move_str)
        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            return "env_board_size=19"

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
