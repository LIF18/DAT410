import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from board import GomokuBoard
from model import GomokuResNet
from mcts_agents import get_action_pure_mcts, get_action_pure_resnet, get_action_neural_mcts, Node

def play_match(agent_black, agent_white, verbose=False):
    """
    Plays a single game between two agents.
    Includes a random opening (first 2 moves) to guarantee a diverse tournament.
    Returns: 1 if Black wins, 2 if White wins, 0 for draw.
    """
    board = GomokuBoard()
    
    # Map player ID to their agent function
    agents = {1: agent_black, 2: agent_white}
    
    move_count = 0
    while not board.is_game_over():
        current_agent = agents[board.current_player]
        
        # Random opening for the first 2 moves (1 for Black, 1 for White)
        if move_count < 2:
            action = random.choice(board.get_legal_moves())
        else:
            action = current_agent(board)

        board.play(action)
        move_count += 1
        
        if verbose:
            print(f"Move {move_count}: Player {3 - board.current_player} played at index {action}")
            board.print_board()
            
    return board.winner

def run_tournament(agent_name_A, agent_func_A, agent_name_B, agent_func_B, num_games=10):
    """
    Runs a tournament between two agents, swapping colors (Black/White) halfway.
    """
    print(f"\n{'='*40}")
    print(f"Tournament: {agent_name_A} vs {agent_name_B} ({num_games} games)")
    print(f"{'='*40}")
    
    wins_A = 0
    wins_B = 0
    draws = 0
    
    for i in tqdm(range(num_games), desc="Playing Games"):
        # Swap colors every game to be fair (Black has first-mover advantage in Gomoku)
        if i % 2 == 0:
            winner = play_match(agent_func_A, agent_func_B, verbose=False)
            if winner == 1: wins_A += 1
            elif winner == 2: wins_B += 1
            else: draws += 1
        else:
            winner = play_match(agent_func_B, agent_func_A, verbose=False)
            if winner == 1: wins_B += 1
            elif winner == 2: wins_A += 1
            else: draws += 1
            
    print(f"\nResults:")
    print(f"{agent_name_A} Wins: {wins_A}")
    print(f"{agent_name_B} Wins: {wins_B}")
    print(f"Draws: {draws}")
    return wins_A, wins_B, draws


def generate_dynamic_heatmap(model, device):
    """
    Runs a self-play match with a random opening to ensure diverse games.
    Dynamically captures a divergence between ResNet's intuition and MCTS's deep search.
    Applies an 'illegal move mask' so the heatmap accurately reflects valid choices.
    """
    print("\nSearching for a perfect divergence example with a random opening")
    board = GomokuBoard()
    
    move_count = 0
    # Let the AI play against itself
    while not board.is_game_over():
        move_count += 1
        
        # Identify current player for context
        current_player_symbol = 'X (Black)' if board.current_player == 1 else 'O (White)'
        legal_moves = board.get_legal_moves()
        
        
        # Random Opening (First 2 moves)
        if move_count <= 2:
            random_action = random.choice(legal_moves)
            board.play(random_action)
            continue
            
   
        # Get ResNet Prior Probabilities (with Legal Masking)
        board_tensor = board.to_tensor().unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(board_tensor)
            raw_probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
            
        # Mask out illegal moves (set probability to 0 for occupied spots)
        masked_resnet_probs = np.zeros(225)
        for m in legal_moves:
            masked_resnet_probs[m] = raw_probs[m]
            
        # Re-normalize so the probabilities of legal moves sum to 1
        if sum(masked_resnet_probs) > 0:
            masked_resnet_probs /= sum(masked_resnet_probs)
            
        # Find the move ResNet greedily prefers among LEGAL moves
        resnet_best_move = max(legal_moves, key=lambda m: masked_resnet_probs[m])
        
        # Run MCTS Validation
        root = Node(None, 1.0)
        num_simulations = 200  
        
        for _ in range(num_simulations):
            node = root
            temp_board = board.copy()
            
            # Selection
            while node.children:
                action, node = max(node.children.items(), key=lambda item: item[1].get_ucb_value())
                temp_board.play(action)
                
            # Expansion & Zero-Rollout Evaluation
            is_terminal = temp_board.is_game_over()
            if not is_terminal:
                l_moves = temp_board.get_legal_moves()
                t_tensor = temp_board.to_tensor().unsqueeze(0).to(device)
                with torch.no_grad():
                    l_probs = model(t_tensor)
                    p = torch.exp(l_probs).squeeze(0).cpu().numpy()
                
                # Apply mask inside MCTS as well
                action_probs = {m: p[m] for m in l_moves}
                s = sum(action_probs.values())
                action_probs = {k: v/s for k, v in action_probs.items()} if s > 0 else {m: 1/len(l_moves) for m in l_moves}
                node.expand(action_probs)
                reward = 0  # Zero-rollout
            else:
                reward = 0 if temp_board.winner == 0 else 1
                
            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.q_value += (reward - node.q_value) / node.visit_count
                node = node.parent
                reward = -reward

        # Find the move with the highest visit count after MCTS
        mcts_visits = np.zeros(225)
        for action, child in root.children.items():
            mcts_visits[action] = child.visit_count
            
        mcts_best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        

        # Golden Divergence Detection & Visualization
        if move_count > 10 and resnet_best_move != mcts_best_move:
            res_y, res_x = resnet_best_move // 15, resnet_best_move % 15
            mcts_y, mcts_x = mcts_best_move // 15, mcts_best_move % 15
            res_coord = f"{chr(ord('a') + res_x)}{res_y + 1}"
            mcts_coord = f"{chr(ord('a') + mcts_x)}{mcts_y + 1}"
            
            print(f"\n Divergence captured at move {move_count} ({current_player_symbol}):")
            print(f"-> Pure ResNet prefers  : {res_coord} (Index {resnet_best_move})")
            print(f"-> Neural MCTS dictates : {mcts_coord} (Index {mcts_best_move})")
            
            board.print_board()

            # Use masked probabilities for the heatmap
            resnet_heatmap = masked_resnet_probs.reshape(15, 15)
            mcts_heatmap = mcts_visits.reshape(15, 15)
            
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            x_ticks = np.arange(15)
            x_labels = [chr(ord('a') + i) for i in range(15)]
            y_ticks = np.arange(15)
            y_labels = [str(i + 1) for i in range(15)]
            
            im1 = axes[0].imshow(resnet_heatmap, cmap='hot', interpolation='nearest')
            axes[0].set_title(f"ResNet Prior P (Masked) | Move {move_count} | {current_player_symbol}")
            axes[0].set_xticks(x_ticks); axes[0].set_xticklabels(x_labels)
            axes[0].set_yticks(y_ticks); axes[0].set_yticklabels(y_labels)
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            
            im2 = axes[1].imshow(mcts_heatmap, cmap='hot', interpolation='nearest')
            axes[1].set_title(f"MCTS Visit Counts ({num_simulations} Sims)")
            axes[1].set_xticks(x_ticks); axes[1].set_xticklabels(x_labels)
            axes[1].set_yticks(y_ticks); axes[1].set_yticklabels(y_labels)
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig("dynamic_evaluation_heatmap1.png", dpi=300, bbox_inches='tight')
            print("Visualization saved successfully!")
            return
            
        board.play(mcts_best_move)

def plot_tournament_results(results_dict):

    # Generates a grouped bar chart summarizing the results of all tournament matchups.
    print("\nGenerating Tournament Results Chart")
    
    matchup_labels = list(results_dict.keys())
    agent1_wins = [res[0] for res in results_dict.values()]
    agent2_wins = [res[1] for res in results_dict.values()]
    draws = [res[2] for res in results_dict.values()]

    x = np.arange(len(matchup_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, agent1_wins, width, label='Agent 1 Wins', color='#d62728')
    rects2 = ax.bar(x, agent2_wins, width, label='Agent 2 Wins', color='#1f77b4')        
    rects3 = ax.bar(x + width, draws, width, label='Draws', color='#7f7f7f')            

    ax.set_ylabel('Number of Games')
    ax.set_title('Tournament Results: Model Comparisons')
    ax.set_xticks(x)
    
    formatted_labels = [label.replace(" vs ", "\nvs\n") for label in matchup_labels]
    ax.set_xticklabels(formatted_labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig('tournament_results_chart.png', dpi=300, bbox_inches='tight')
    print("Chart saved successfully as 'tournament_results_chart.png'")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to {device}")
    
    # Initialize model and load trained weights
    model = GomokuResNet(num_blocks=5, num_filters=64).to(device)
    
    try:
        model.load_state_dict(torch.load("best_resnet_gomoku.pth", map_location=device))
        model.eval()
        print("Model weights loaded successfully!")
    except FileNotFoundError:
        print("WARNING: 'best_resnet_gomoku.pth' not found. Ensure train.py has finished.")
        exit(1)


    # Define Agent Wrappers (to match the (board) -> action signature)
    # Note: Pure MCTS is very slow on 15x15. We keep simulations low.

    def agent_pure_mcts(board):
        # 100 simulations is low, but enough to show it's worse than Neural-Guided
        return get_action_pure_mcts(board, num_simulations=100) 
        
    def agent_pure_resnet(board):
        return get_action_pure_resnet(board, model, device)
        
    def agent_neural_mcts(board):
        return get_action_neural_mcts(board, model, device, num_simulations=100)

    tournament_results = {}

    # Run Tournament 1: Pure MCTS vs Neural-Guided MCTS
    w1, w2, d = run_tournament("Pure MCTS", agent_pure_mcts, 
                   "Neural MCTS", agent_neural_mcts, 
                   num_games=10)
    tournament_results["Pure MCTS vs Neural MCTS"] = (w1, w2, d)

    # Run Tournament 2: Pure ResNet vs Neural-Guided MCTS
    w1, w2, d = run_tournament("Pure ResNet", agent_pure_resnet, 
                   "Neural MCTS", agent_neural_mcts, 
                   num_games=10)
    tournament_results["Pure ResNet vs Neural MCTS"] = (w1, w2, d)

    # Run Tournament 3: Pure ResNet vs Pure MCTS
    w1, w2, d = run_tournament("Pure ResNet", agent_pure_resnet,
                   "Pure MCTS", agent_pure_mcts,
                   num_games=10)
    tournament_results["Pure ResNet vs Pure MCTS"] = (w1, w2, d)
    plot_tournament_results(tournament_results)

    # Generate Heatmap Visualization
    generate_dynamic_heatmap(model, device)
    
    print("\nAll evaluations complete")