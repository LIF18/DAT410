import numpy as np
import torch
import random

class GomokuBoard:
    def __init__(self):
        # Board state: 0 represents empty, 1 represents black, 2 represents white
        self.size = 15
        self.state = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1  # 1 (Black) goes first
        self.last_move = None    # Record the last move (1D index: 0~224)
        self.winner = None       # 1: Black wins, 2: White wins, 0: Draw, None: Game in progress
        self.availables = list(range(self.size * self.size)) # Remaining available empty spots

    def copy(self):
        """
        Deep copy the current board. MCTS needs to frequently copy the board 
        when exploring tree nodes without polluting the real game.
        """
        new_board = GomokuBoard()
        new_board.state = np.copy(self.state)
        new_board.current_player = self.current_player
        new_board.last_move = self.last_move
        new_board.winner = self.winner
        new_board.availables = self.availables.copy()
        return new_board

    def get_legal_moves(self):
        # Return all current legal move positions (1D indices from 0 to 224)
        return self.availables

    def play(self, action):
        """
        Play a move at the specified position and update the game state
        :param action: 1D integer index between 0 and 224
        """
        if self.winner is not None:
            return  # Game is over, cannot play anymore

        # Convert 1D index to 2D coordinates
        y = action // self.size
        x = action % self.size

        if self.state[y, x] != 0:
            raise ValueError(f"Position {action} (y={y}, x={x}) already has a piece!")

        # Play the piece
        self.state[y, x] = self.current_player
        self.last_move = action
        self.availables.remove(action)

        # Check if this move ends the game
        if self._check_win(y, x, self.current_player):
            self.winner = self.current_player
        elif len(self.availables) == 0:
            self.winner = 0  # Draw

        # Switch player
        self.current_player = 2 if self.current_player == 1 else 1

    def is_game_over(self):
        # Check if the game is over
        return self.winner is not None

    def _check_win(self, y, x, player):
        """
        Highly efficient win/loss judgment: only checks the 4 directions around 
        the just-played (y, x) to see if there are 5 consecutive pieces of the same color.
        """
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Main diagonal \
            (1, -1)   # Anti-diagonal /
        ]

        for dy, dx in directions:
            count = 1  # Including the piece just played
            
            # Forward exploration
            ny, nx = y + dy, x + dx
            while 0 <= ny < self.size and 0 <= nx < self.size and self.state[ny, nx] == player:
                count += 1
                ny += dy
                nx += dx
                
            # Backward exploration
            ny, nx = y - dy, x - dx
            while 0 <= ny < self.size and 0 <= nx < self.size and self.state[ny, nx] == player:
                count += 1
                ny -= dy
                nx -= dx

            if count >= 5:
                return True
                
        return False

    def simulate_random_game_to_end(self):
        """
        Simulation phase of MCTS: starting from the current board, both sides 
        play randomly until a winner emerges.
        """
        temp_board = self.copy()
        
        # Record from whose perspective the Reward is calculated
        view_player = temp_board.current_player

        # Random playouts until the end
        while not temp_board.is_game_over():
            random_action = random.choice(temp_board.get_legal_moves())
            temp_board.play(random_action)

        # Calculate and return reward
        if temp_board.winner == 0:
            return 0  # Draw
        elif temp_board.winner == view_player:
            return 1  # Won
        else:
            return -1 # Lost

    def to_tensor(self):
        """
        Convert the board state to a PyTorch Tensor and feed it to ResNet for evaluation.
        Output shape: (3, 15, 15)
        Channel 0: Black (1)
        Channel 1: White (2)
        Channel 2: Empty (0)
        """
        board_tensor = torch.zeros((3, self.size, self.size), dtype=torch.float32)
        
        # Extract boolean masks for each state
        black_mask = (self.state == 1)
        white_mask = (self.state == 2)
        empty_mask = (self.state == 0)

        # Fill into the corresponding channels
        board_tensor[0][black_mask] = 1.0
        board_tensor[1][white_mask] = 1.0
        board_tensor[2][empty_mask] = 1.0

        return board_tensor

    def print_board(self):
        # Used to print the current board in the console terminal for easy visualization of the match process
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print("  " + " ".join([chr(ord('a') + i) for i in range(self.size)]))
        for y in range(self.size):
            row_str = f"{y+1:2d} "
            for x in range(self.size):
                row_str += symbols[self.state[y, x]] + " "
            print(row_str)
        print("-------------------------------")