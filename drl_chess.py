import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ==========================================
# 1. Environment: Wrapper around python-chess
# ==========================================
class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.get_state()

    def step(self, action_uci):
        action = chess.Move.from_uci(action_uci)
        if action in self.board.legal_moves:
            self.board.push(action)
        else:
            raise ValueError(f"Illegal move: {action_uci}")
        
        done = self.board.is_game_over()
        reward = self.get_reward() if done else 0
        return self.get_state(), reward, done

    def get_reward(self):
        result = self.board.result()
        if result == '1-0':
            return 1 # White wins
        elif result == '0-1':
            return -1 # Black wins
        return 0 # Draw

    def get_state(self):
        return self.board.copy()
    
    def get_legal_moves(self):
        return [move.uci() for move in self.board.legal_moves]

# ==========================================
# 2. State Encoder: Tensor conversion
# ==========================================
def encode_board(board):
    """
    Converts a chess.Board to an 8x8x14 tensor.
    Channels: 12 for pieces (6 white, 6 black), 1 for player turn, 1 for castling/en-passant flags (simplified).
    """
    state = np.zeros((14, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    
    for square, piece in piece_map.items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        
        # Piece types: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
        channel = piece.piece_type - 1
        if not piece.color: # Black pieces shifted by 6
            channel += 6
            
        state[channel, row, col] = 1.0

    # 13th channel: turn
    state[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    # 14th channel: can be used for castling rights (simplified here to 1)
    state[13, :, :] = 1.0 
    
    return torch.tensor(state).unsqueeze(0) # Add batch dimension

# ==========================================
# 3. Policy and Value Networks
# ==========================================
class DRLChessNet(nn.Module):
    def __init__(self):
        super(DRLChessNet, self).__init__()
        
        # Shared Convolutional Layers
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Policy Head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096) # Simplified action space size
        
        # Value Head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = F.log_softmax(self.policy_fc(p), dim=1) # Log probabilities
        
        # Value
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # Output between -1 and 1
        
        return p, v

# ==========================================
# 4. Action Selector (MCTS / Epsilon-Greedy)
# ==========================================
def move_to_index(move_uci):
    # A simple 4096 mapping: from_square * 64 + to_square
    # Ignore promotion character for index to keep it size 4096
    move = chess.Move.from_uci(move_uci)
    return move.from_square * 64 + move.to_square

class ActionSelector:
    def __init__(self, method="epsilon-greedy", epsilon=0.1):
        self.method = method
        self.epsilon = epsilon

    def select_action(self, env, network, device):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None

        if self.method == "epsilon-greedy":
            if random.random() < self.epsilon:
                return random.choice(legal_moves)
            else:
                # Greedy selection using policy network
                state_tensor = encode_board(env.board).to(device)
                with torch.no_grad():
                    policy, _ = network(state_tensor)
                
                policy = policy.squeeze(0) # Shape: [4096]
                
                best_move = None
                best_prob = -float('inf')
                for move in legal_moves:
                    idx = move_to_index(move)
                    prob = policy[idx].item()
                    if prob > best_prob:
                        best_prob = prob
                        best_move = move
                        
                return best_move if best_move else random.choice(legal_moves)
                
        elif self.method == "mcts":
            import math
            class MCTSNode:
                def __init__(self, prior):
                    self.prior = prior
                    self.visit_count = 0
                    self.value_sum = 0
                    self.children = {}
                def q_value(self):
                    return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

            root = MCTSNode(1.0)
            state_tensor = encode_board(env.board).to(device)
            with torch.no_grad():
                policy, _ = network(state_tensor)
            policy = policy.squeeze(0)
            
            for move in legal_moves:
                prob = torch.exp(policy[move_to_index(move)]).item()
                root.children[move] = MCTSNode(prior=prob)
                
            num_simulations = 40 # MCTS推演次數 (AlphaZero用800，但CPU太慢所以用40)
            for _ in range(num_simulations):
                node = root
                sim_env = ChessEnv()
                sim_env.board = env.board.copy()
                path = [node]
                
                # 1. Selection
                while node.children:
                    turn = sim_env.board.turn
                    best_score = -float('inf')
                    best_action = None
                    best_child = None
                    for action, child in node.children.items():
                        u = 1.0 * child.prior * math.sqrt(node.visit_count + 1e-8) / (1 + child.visit_count)
                        q = child.q_value()
                        if turn == chess.BLACK:
                            q = -q # 黑方要最小化白方勝率
                        if q + u > best_score:
                            best_score = q + u
                            best_action = action
                            best_child = child
                    node = best_child
                    path.append(node)
                    sim_env.step(best_action)
                    
                # 2. Expansion & Evaluation
                done = sim_env.board.is_game_over()
                if not done:
                    sim_tensor = encode_board(sim_env.board).to(device)
                    with torch.no_grad():
                        sim_policy, value = network(sim_tensor)
                    sim_policy = sim_policy.squeeze(0)
                    value = value.item()
                    
                    for move in sim_env.get_legal_moves():
                        prob = torch.exp(sim_policy[move_to_index(move)]).item()
                        node.children[move] = MCTSNode(prior=prob)
                else:
                    value = sim_env.get_reward()
                    
                # 3. Backpropagation
                for n in path:
                    n.visit_count += 1
                    n.value_sum += value
                    
            best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
            return best_move

# ==========================================
# 5. Self-Play Loop
# ==========================================
def self_play(network, episodes=10, selector_method="epsilon-greedy", device='cpu'):
    env = ChessEnv()
    selector = ActionSelector(method=selector_method)
    
    memory = deque(maxlen=10000)
    
    for episode in range(episodes):
        env.reset()
        done = False
        trajectory = []
        
        print(f"Starting Episode {episode+1}")
        while not done:
            state_copy = env.get_state()
            action = selector.select_action(env, network, device)
            
            if action is None:
                break
                
            # Set the chosen action to 1 in pi for training
            pi = np.zeros(4096) 
            pi[move_to_index(action)] = 1.0
            trajectory.append((state_copy, pi))
            
            _, reward, done = env.step(action)
            
        print(f"Episode {episode+1} finished. Result: {env.board.result()}")
        
        # Assign rewards to the trajectory
        for state, pi in trajectory:
            # If black played, reward is inverted from the perspective of white
            z = reward if state.turn == chess.WHITE else -reward
            memory.append((state, pi, z))
            
    return memory

# ==========================================
# 6. Training Module
# ==========================================
def train(network, memory, epochs=1, batch_size=32, device='cpu'):
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        if len(memory) < batch_size:
            continue
            
        batch = random.sample(memory, batch_size)
        
        states = []
        pis = []
        zs = []
        
        for state, pi, z in batch:
            states.append(encode_board(state).squeeze(0))
            pis.append(torch.tensor(pi, dtype=torch.float32))
            zs.append(torch.tensor([z], dtype=torch.float32))
            
        state_tensor = torch.stack(states).to(device)
        pi_tensor = torch.stack(pis).to(device)
        z_tensor = torch.stack(zs).to(device)
        
        optimizer.zero_grad()
        
        p, v = network(state_tensor)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(v, z_tensor)
        
        # Policy loss (Policy Gradient / REINFORCE)
        # 這裡非常重要！我們必須乘上勝負回饋 (z_tensor)
        # 這樣網路才會知道：「贏了」要強化這個走法，「輸了」要避免這個走法
        policy_loss = -torch.sum(pi_tensor * p * z_tensor) / batch_size
        
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} (V: {value_loss.item():.4f}, P: {policy_loss.item():.4f})")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize network
    net = DRLChessNet().to(device)
    
    # --- 階段三：迭代進化 (Iteration) ---
    iterations = 5  # 總共進行 5 大輪的進化
    episodes_per_iter = 5  # MCTS 運算量非常龐大，為了能在 CPU 上跑完，先調低至 5 局
    epochs_per_iter = 5    # 每輪收集完經驗後，讓大腦訓練 5 次
    
    for iteration in range(iterations):
        print(f"\n===========================================")
        print(f"   開始第 {iteration+1}/{iterations} 輪迭代進化")
        print(f"===========================================")
        
        # 1. 產生經驗 (Self-Play)
        print("--- 階段一：自我對弈收集資料 (使用 MCTS) ---")
        memory = self_play(net, episodes=episodes_per_iter, selector_method="mcts", device=device)
        
        # 2. 學習經驗 (Training)
        print("--- 階段二：神經網路學習 ---")
        # 這裡會拿剛剛下棋的 memory 來更新網路 (net) 的權重
        train(net, memory, epochs=epochs_per_iter, batch_size=16, device=device)
        
        # 3. 儲存進化的結果
        torch.save(net.state_dict(), "model.pth")
        print(f"第 {iteration+1} 輪進化完成！模型已儲存至 model.pth")
        
    print("\nDRL_Chess 所有訓練迭代已順利完成！")
