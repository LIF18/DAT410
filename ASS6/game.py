import math
import random

# Game Environment
class TicTacToe:
    def __init__(self, state=None, player=1):
        self.state = state if state else [0]*9
        self.player = player # 1 for X (Human), -1 for O (MCTS)

    def get_legal_moves(self):
        if self.check_winner() is not None:
            return []
        return [i for i, x in enumerate(self.state) if x == 0]

    def make_move(self, move):
        new_state = self.state.copy()
        new_state[move] = self.player
        return TicTacToe(new_state, -self.player)

    def check_winner(self):
        winning_positions = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in winning_positions:
            if self.state[a] != 0 and self.state[a] == self.state[b] == self.state[c]:
                return self.state[a]
        if 0 not in self.state:
            return 0 # Draw
        return None # Game ongoing

    def print_board(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("\nBoard:")
        for i in range(3):
            row = [symbols[self.state[i*3 + j]] for j in range(3)]
            print(f" {row[0]} | {row[1]} | {row[2]} ")
            if i < 2:
                print("---+---+---")
        print()

# MCTS Algorithm
class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = game_state.get_legal_moves()

    def uct_select_child(self, c=math.sqrt(2)):
        return max(self.children, key=lambda n: n.wins/n.visits + c * math.sqrt(math.log(self.visits)/n.visits))

    def expand(self):
        move = self.untried_moves.pop()
        next_state = self.game_state.make_move(move)
        child_node = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        # Standard MCTS scoring: Win=1, Draw=0.5, Loss=0
        if result == -self.game_state.player:
             self.wins += 1    # The player who made the move won
        elif result == 0:
             self.wins += 0.5  # Draw
        # If it is a loss, we simply add 0

        if self.parent:
            self.parent.backpropagate(result)
        # if result == self.game_state.player:
        #      self.wins -= 1 
        # elif result == -self.game_state.player:
        #      self.wins += 1
        # elif result == 0:
        #      self.wins += 0.5
        # if self.parent:
        #     self.parent.backpropagate(result)

def mcts(root_state, iterations=2000):
    root = MCTSNode(root_state)
    for _ in range(iterations):
        node = root
        # Selection
        while not node.untried_moves and node.children:
            node = node.uct_select_child()
        # Expansion
        if node.untried_moves:
            node = node.expand()
        # Rollout
        current_state = node.game_state
        while current_state.check_winner() is None:
            move = random.choice(current_state.get_legal_moves())
            current_state = current_state.make_move(move)
        # Backpropagate
        node.backpropagate(current_state.check_winner())
    return max(root.children, key=lambda n: n.visits).move


# Interactive Game Loop
def play_game():
    game = TicTacToe()
    print("Welcome to Tic-Tac-Toe! (Human vs MCTS)")
    print("Positions are 0-8, corresponding to:")
    print(" 0 | 1 | 2 \n---+---+---\n 3 | 4 | 5 \n---+---+---\n 6 | 7 | 8 \n")
    
    while game.check_winner() is None:
        game.print_board()
        
        if game.player == 1:
            # Human's turn
            valid_move = False
            while not valid_move:
                try:
                    move = int(input("Enter your move (0-8): "))
                    if move in game.get_legal_moves():
                        valid_move = True
                    else:
                        print("Invalid move. Cell is already taken or out of range.")
                except ValueError:
                    print("Please enter a number between 0 and 8.")
            print(f"Human (X) played position {move}")
            game = game.make_move(move)
            
        else:
            # MCTS Agent's turn
            print("MCTS Agent (O) is thinking...")
            # We use 2000 iterations to ensure it plays perfectly
            move = mcts(game, iterations=2000)
            print(f"MCTS Agent (O) played position {move}")
            game = game.make_move(move)

    # Game Over
    game.print_board()
    winner = game.check_winner()
    if winner == 1:
        print("Result: Human (X) wins! (Wait, that's impossible against 2000-iteration MCTS!)")
    elif winner == -1:
        print("Result: MCTS Agent (O) wins!")
    else:
        print("Result: It's a Draw!")

if __name__ == "__main__":
    play_game()