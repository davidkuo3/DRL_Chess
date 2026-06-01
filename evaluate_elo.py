import chess
import torch
import random
import math
import os
from drl_chess import ChessEnv, DRLChessNet, ActionSelector

def play_match(model1, model2, device, games=10):
    """
    讓兩個模型對弈指定的局數，並計算 model1 的勝率
    (平手算 0.5 勝)
    """
    env = ChessEnv()
    score_1 = 0
    
    for i in range(games):
        env.reset()
        done = False
        
        # 交換黑白方，確保公平
        if i % 2 == 0:
            white, black = model1, model2
        else:
            white, black = model2, model1
            
        print(f"Game {i+1}/{games} | {white['name']} (白) vs {black['name']} (黑)")
        
        step_count = 0
        while not done:
            current_player = white if env.board.turn == chess.WHITE else black
            
            if current_player['name'] == 'Random Agent':
                action = random.choice(env.get_legal_moves())
            else:
                action, _ = current_player['selector'].select_action(env, current_player['net'], device)
                
            env.step(action)
            done = env.board.is_game_over()
            step_count += 1
            
            # 避免對局陷入無窮迴圈（例如雙方都不吃子）
            if step_count > 200:
                print("對局超過 200 步，強制以和局結束。")
                break
            
        if step_count <= 200:
            result = env.board.result()
        else:
            result = '1/2-1/2'
            
        print(f"--> 結果: {result} (共 {step_count} 步)")
        
        if result == '1-0':
            if white == model1: score_1 += 1
        elif result == '0-1':
            if black == model1: score_1 += 1
        else: # 和局 (Draw)
            score_1 += 0.5
            
    return score_1 / games

def calculate_elo(win_rate, opponent_elo=100):
    """
    根據標準 Elo 公式計算 Elo 分數。
    E_A = 1 / (1 + 10^((R_B - R_A)/400))
    """
    # 避免對數計算無限大，將勝率限制在 0.01 到 0.99 之間
    win_rate = max(0.01, min(0.99, win_rate))
    return opponent_elo - 400 * math.log10(1 / win_rate - 1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 載入我們訓練好的模型
    net = DRLChessNet().to(device)
    if os.path.exists("model.pth"):
        net.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
        print("✅ 成功載入 trained model.pth")
    else:
        print("⚠️ 未找到 model.pth，將測試『未經訓練的隨機權重神經網路』")
    
    net.eval()
    
    # 建立對戰雙方
    model_drl = {
        'name': 'DRL Model (MCTS)',
        'net': net,
        'selector': ActionSelector(method="mcts")
    }
    
    model_random = {
        'name': 'Random Agent',
        'net': None,
        'selector': None
    }
    
    # 設定測試局數 (局數越多，評估越準，但花費時間越長)
    num_games = 10
    
    print("\n===========================================")
    print("   開始 Elo 評分測試 (vs 隨機猴子)")
    print("===========================================")
    
    win_rate = play_match(model_drl, model_random, device, games=num_games)
    
    print("\n===========================================")
    print("   評分結果")
    print("===========================================")
    print(f"對戰隨機對手勝率: {win_rate*100:.1f}%")
    
    # 假設完全隨機亂走的對手 Elo 為 100
    estimated_elo = calculate_elo(win_rate, opponent_elo=100)
    print(f"估計 Elo 積分: {estimated_elo:.0f}")
    
    if estimated_elo > 100:
        print("🎉 恭喜！你的模型表現已經超越了隨機猴子，確實有學到東西！")
    else:
        print("繼續努力！模型目前表現還不如完全隨機亂走。")
