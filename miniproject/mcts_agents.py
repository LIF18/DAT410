import torch
import math
import numpy as np

from model import GomokuResNet
from board import GomokuBoard  


class Node:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {} # key: action, value: Node
        self.visit_count = 0 
        self.q_value = 0     
        self.prior_p = prior_p 

    def expand(self, action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def get_ucb_value(self, c_puct=1.5):
        # Avoid division by 0, add a small constant or ensure 1 + visit_count
        u = c_puct * self.prior_p * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value + u


# Model A: Pure MCTS
def get_action_pure_mcts(board, num_simulations=1000):
    root = Node(None, 1.0)
    for _ in range(num_simulations):
        node = root
        temp_board = board.copy()
        
        while node.children:
            action, node = max(node.children.items(), key=lambda item: item[1].get_ucb_value())
            temp_board.play(action)
            
        legal_moves = temp_board.get_legal_moves()
        if not temp_board.is_game_over():
            uniform_prob = 1.0 / len(legal_moves)
            action_probs = {move: uniform_prob for move in legal_moves}
            node.expand(action_probs)
            
        reward = temp_board.simulate_random_game_to_end()
        
        while node is not None:
            node.visit_count += 1
            node.q_value += (reward - node.q_value) / node.visit_count
            node = node.parent
            reward = -reward 
            
    best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
    return best_action


# Model B: Pure ResNet
def get_action_pure_resnet(board, model, device):
    board_tensor = board.to_tensor().unsqueeze(0).to(device)
    
    with torch.no_grad():
        log_probs = model(board_tensor)
        probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
        
    legal_moves = board.get_legal_moves()
    # Find the maximum probability only among legal moves
    legal_probs = [(move, probs[move]) for move in legal_moves]
    best_action = max(legal_probs, key=lambda x: x[1])[0]
    
    return best_action


# Model C: Neural-Guided MCTS (Fixed version: Zero-Rollout)
def get_action_neural_mcts(board, model, device, num_simulations=100):
    root = Node(None, 1.0)
    for _ in range(num_simulations):
        node = root
        temp_board = board.copy()
        
        # 1. Selection
        while node.children:
            action, node = max(node.children.items(), key=lambda item: item[1].get_ucb_value())
            temp_board.play(action)
            
        # 2. Expansion 
        is_terminal = temp_board.is_game_over()
        
        if not is_terminal:
            # If not finished, use ResNet to output the probabilities of subsequent actions
            legal_moves = temp_board.get_legal_moves()
            board_tensor = temp_board.to_tensor().unsqueeze(0).to(device)
            with torch.no_grad():
                log_probs = model(board_tensor)
                probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
            
            action_probs = {move: probs[move] for move in legal_moves}
            sum_probs = sum(action_probs.values())
            if sum_probs > 0:
                action_probs = {k: v / sum_probs for k, v in action_probs.items()}
            else:
                action_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
                
            node.expand(action_probs)
            
            # Core Modification: No longer perform simulate_random_game_to_end()
            # Since we don't have a separate Value Network, we directly assume the current situation is unclear and return a reward of 0.
            # This makes MCTS fully rely on the P value (ResNet's judgment) in the UCB formula to expand,
            # but it can use the tree structure to foresee the opponent's fatal moves like "open three/four in a row" in advance.
            reward = 0  
        else:
            # If the game ends during MCTS exploration (a move directly connects 5)
            # We give a large real reward, which helps MCTS learn "one-hit kill" or "desperate defense".
            if temp_board.winner == 0:
                reward = 0 # Draw
            else:
                # The player who just moved won, so from the current node's perspective, the previous move was excellent
                reward = 1 
        
        # 3. Backpropagation
        while node is not None:
            node.visit_count += 1
            node.q_value += (reward - node.q_value) / node.visit_count
            node = node.parent
            reward = -reward  # Reward flips each time we move up a level (the opponent's poison is my honey)

    best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
    return best_action



# Model C: Neural-Guided MCTS
# def get_action_neural_mcts(board, model, device, num_simulations=400):
#     root = Node(None, 1.0)
#     for _ in range(num_simulations):
#         node = root
#         temp_board = board.copy()
        
#         while node.children:
#             action, node = max(node.children.items(), key=lambda item: item[1].get_ucb_value())
#             temp_board.play(action)
            
#         legal_moves = temp_board.get_legal_moves()
#         if not temp_board.is_game_over():
#             board_tensor = temp_board.to_tensor().unsqueeze(0).to(device)
#             with torch.no_grad():
#                 log_probs = model(board_tensor)
#                 probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
            
#             action_probs = {move: probs[move] for move in legal_moves}
#             sum_probs = sum(action_probs.values())
#             if sum_probs > 0:
#                 action_probs = {k: v / sum_probs for k, v in action_probs.items()}
#             else:
#                 action_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
                
#             node.expand(action_probs)
            
#         reward = temp_board.simulate_random_game_to_end()
        
#         while node is not None:
#             node.visit_count += 1
#             node.q_value += (reward - node.q_value) / node.visit_count
#             node = node.parent
#             reward = -reward

#     best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
#     return best_action