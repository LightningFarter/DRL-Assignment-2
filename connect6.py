import sys
import numpy as np
import random
import math

class SimConnect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
    
    def copy(self):
        new_game = SimConnect6Game(self.size)
        new_game.board = self.board.copy()
        new_game.turn = self.turn
        new_game.game_over = self.game_over
        return new_game

    def play_move(self, move_str, color):
        if self.game_over:
            return True
        stones = move_str.split(',')
        positions = []
        for stone in stones:
            stone = stone.strip().upper()
            if len(stone) < 1:
                return False
            col_char = stone[0]
            row_str = stone[1:]
            try:
                row = int(row_str) - 1
                col = self.label_to_index(col_char)
            except ValueError:
                return False
            if row < 0 or row >= self.size or col < 0 or col >= self.size:
                return False
            if self.board[row, col] != 0:
                return False
            positions.append((row, col))
        player = 1 if color == 'B' else 2
        for row, col in positions:
            self.board[row, col] = player
        self.turn = 3 - self.turn
        winner = self.check_win()
        if winner != 0:
            self.game_over = True
        return self.game_over
    
    def check_win(self):
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

    def label_to_index(self, col_char):
        col_char = col_char.upper()
        if col_char >= 'J':
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def index_to_label(self, col):
        return chr(ord('A') + col + (1 if col >= 8 else 0))


class Node:
    def __init__(self, parent=None, state=None, player=None):
        self.parent = parent
        self.state = state
        self.player = player
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
    
    def is_fully_expanded(self):
        return False


class MCTS:
    def __init__(self, root_state, root_player, exploration_weight=1.414):
        self.root = Node(state=root_state, player=root_player)
        self.exploration_weight = exploration_weight
    
    def select_node(self):
        current_node = self.root
        while current_node.children:
            if not current_node.is_fully_expanded():
                return current_node
            best_score = -float('inf')
            best_child = None
            for child in current_node.children.values():
                if child.visit_count == 0:
                    score = float('inf')
                else:
                    exploitation = child.total_value / child.visit_count
                    exploration = math.sqrt(math.log(current_node.visit_count) / child.visit_count)
                    score = exploitation + self.exploration_weight * exploration
                if score > best_score:
                    best_score = score
                    best_child = child
            current_node = best_child
        return current_node
    
    def expand(self, node: Node):
        if node.state.game_over:
            return None
        n = 1 if np.all(node.state.board == 0) else 2
        empty_positions = list(zip(*np.where(node.state.board == 0)))
        if len(empty_positions) < n:
            return None
        for _ in range(100):
            selected = random.sample(empty_positions, n)
            if n == 2:
                selected = sorted(selected, key=lambda pos: (pos[0], pos[1]))
            move_parts = []
            for (r, c) in selected:
                col_label = node.state.index_to_label(c)
                move_part = f"{col_label}{r+1}"
                move_parts.append(move_part)
            move_str = ','.join(move_parts)
            if move_str not in node.children:
                new_state = node.state.copy()
                color = 'B' if node.player == 1 else 'W'
                success = new_state.play_move(move_str, color)
                if not success:
                    continue
                new_player = 3 - node.player
                child_node = Node(parent=node, state=new_state, player=new_player)
                node.children[move_str] = child_node
                return child_node
        return None

    def simulate(self, node: Node):
        sim_state = node.state.copy()
        current_player = node.player
        steps = 0
        while not sim_state.game_over and steps < 100:
            n = 1 if np.all(sim_state.board == 0) else 2
            empty_positions = list(zip(*np.where(sim_state.board == 0)))
            if len(empty_positions) < n:
                break
            selected = random.sample(empty_positions, n)
            move_parts = []
            for (r, c) in selected:
                col_label = sim_state.index_to_label(c)
                move_part = f"{col_label}{r+1}"
                move_parts.append(move_part)
            move_str = ",".join(move_parts)
            color = 'B' if current_player == 1 else 'W'
            sim_state.play_move(move_str, color)
            current_player = 3 - current_player
            steps += 1
        winner = sim_state.check_win()
        if winner == 0:
            return 0.5
        elif winner == self.root.player:
            return 1.0
        else:
            return 0.0

    def backpropagate(self, node: Node, result):
        while node is not None:
            node.visit_count += 1
            node.total_value += result
            node = node.parent
    
    def best_action(self):
        if not self.root.children:
            return None
        best_visit = -1
        best_action = None
        for action, child in self.root.children.items():
            if child.visit_count > best_visit:
                best_visit = child.visit_count
                best_action = action
        return best_action


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
        
        root_state = SimConnect6Game(self.size)
        root_state.board = self.board.copy()
        root_state.turn = self.turn
        root_state.game_over = self.game_over
        root_player = 1 if color.upper() == 'B' else 2

        mcts = MCTS(root_state, root_player)
        for _ in range(500):  # Number of MCTS iterations
            node = mcts.select_node()
            if node.state.game_over:
                winner = node.state.check_win()
                result = 1.0 if winner == root_player else 0.0 if winner != 0 else 0.5
                mcts.backpropagate(node, result)
                continue
            child = mcts.expand(node)
            if child is not None:
                result = mcts.simulate(child)
                mcts.backpropagate(child, result)

        best_action = mcts.best_action()
        if best_action is None:
            print("? No valid move")
            return

        self.play_move(color, best_action)
        print(f"{best_action}\n\n", end='', flush=True)
        print(best_action, file=sys.stderr)

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
