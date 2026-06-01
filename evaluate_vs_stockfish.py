import os
import chess
import torch
import numpy as np
from drl_chess import DRLChessNet, ActionSelector, encode_board, ChessEnv
from stockfish import Stockfish

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STOCKFISH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "stockfish", "stockfish-windows-x86-64-avx2.exe"))
# ---------------------------------------------------------------------------
# Stockfish Elo configuration – we will test several strengths via SkillLevel (0‑20).
# Mapping approximate Elo → SkillLevel (official Stockfish mapping)
ELO_SKILL_MAP = {
    1350: 0, 1500: 2, 1650: 4, 1800: 6, 1950: 8,
    2100: 10, 2250: 12, 2400: 14, 2550: 16,
    2700: 17, 2850: 18, 3000: 19, 3400: 20,
}
# List of target Elo values we want to evaluate against.
TARGET_ELOS = [1500, 2000, 2500, 3000]

STOCKFISH_DEPTH = 8               # search depth for Stockfish (adjustable)
NUM_GAMES = 10                    # total games to play (will be split equally for colors)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Helper: run a single game between DRL agent (MCTS) and Stockfish
# ---------------------------------------------------------------------------
def play_game(model, env, stockfish, drl_color: chess.Color):
    env.reset()
    while not env.board.is_game_over():
        turn = env.board.turn
        if turn == drl_color:
            # DRL move (MCTS)
            selector = ActionSelector(method="mcts")
            move = selector.select_action(env, model, DEVICE)[0]
        else:
            # Stockfish move
            board_fen = env.board.fen()
            stockfish.set_fen_position(board_fen)
            move = stockfish.get_best_move()
        env.step(move)
    # Result from the perspective of DRL (white positive)
    result = env.board.result()
    if result == "1-0":
        return 1 if drl_color == chess.WHITE else -1
    elif result == "0-1":
        return -1 if drl_color == chess.WHITE else 1
    else:
        return 0

# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------
def main():
    # Load trained model (if exists)
    model = DRLChessNet().to(DEVICE)
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=DEVICE, weights_only=True))
        print("Loaded trained model from model.pth")
    else:
        print("No trained model found – exiting.")
        return

    # ---------------------------------------------------------------------
    # Iterate over each target Elo, configure Stockfish, and evaluate.
    # ---------------------------------------------------------------------
    for target_elo in TARGET_ELOS:
        # Find closest skill level from the map
        skill_level = min(ELO_SKILL_MAP, key=lambda k: abs(k - target_elo))
        skill = ELO_SKILL_MAP[skill_level]
        try:
            stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": 2})
            stockfish.set_skill_level(skill)
            # Optionally enforce depth (already set globally via STOCKFISH_DEPTH)
            print(f"\n=== Evaluating vs Stockfish ~{skill_level} Elo (SkillLevel={skill}) ===")
        except Exception as e:
            print(f"Failed to start Stockfish engine: {e}")
            return

        env = ChessEnv()
        drl_wins = 0
        stockfish_wins = 0
        draws = 0

        for i in range(NUM_GAMES):
            drl_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            result = play_game(model, env, stockfish, drl_color)
            if result == 1:
                drl_wins += 1
            elif result == -1:
                stockfish_wins += 1
            else:
                draws += 1
            print(f"Game {i+1}/{NUM_GAMES}: {'DRL' if result==1 else ('Stockfish' if result==-1 else 'Draw')}")

        total = drl_wins + stockfish_wins + draws
        win_rate = drl_wins / total if total > 0 else 0.0
        # Assume Stockfish baseline Elo ≈ target_elo (approximation)
        if win_rate == 0:
            drl_elo = 0
        elif win_rate == 1:
            drl_elo = target_elo + 400
        else:
            drl_elo = target_elo - 400 * (np.log10(1 / win_rate - 1))
        print("--- Evaluation Summary ---")
        print(f"Target Stockfish Elo ≈ {target_elo}, SkillLevel={skill}")
        print(f"DRL wins: {drl_wins}, Stockfish wins: {stockfish_wins}, draws: {draws}")
        print(f"DRL win‑rate vs Stockfish: {win_rate:.3%}")
        print(f"Estimated DRL Elo (relative to Stockfish {target_elo}): {drl_elo:.0f}\n")

    print("\nAll evaluations completed.")


    # Initialise Stockfish engine
    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, parameters={"Threads": 2})
        stockfish.set_depth(STOCKFISH_DEPTH)
    except Exception as e:
        print(f"Failed to start Stockfish engine: {e}")
        return
    print(f"Stockfish engine loaded (depth={STOCKFISH_DEPTH})")

    env = ChessEnv()
    drl_wins = 0
    stockfish_wins = 0
    draws = 0

    for i in range(NUM_GAMES):
        drl_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        result = play_game(model, env, stockfish, drl_color)
        if result == 1:
            drl_wins += 1
        elif result == -1:
            stockfish_wins += 1
        else:
            draws += 1
        print(f"Game {i+1}/{NUM_GAMES}: {'DRL' if result==1 else ('Stockfish' if result==-1 else 'Draw')}")

    total = drl_wins + stockfish_wins + draws
    win_rate = drl_wins / total
    # Assume Stockfish ~ 3500 Elo (very strong). Compute DRL Elo via standard formula.
    stockfish_elo = 3500
    if win_rate == 0:
        drl_elo = 0
    elif win_rate == 1:
        drl_elo = stockfish_elo + 400
    else:
        drl_elo = stockfish_elo - 400 * (np.log10(1 / win_rate - 1))
    print("--- Evaluation Summary ---")
    print(f"DRL wins: {drl_wins}, Stockfish wins: {stockfish_wins}, draws: {draws}")
    print(f"DRL win rate vs Stockfish: {win_rate:.3f}")
    print(f"Estimated DRL Elo (relative to Stockfish {stockfish_elo}): {drl_elo:.0f}")

if __name__ == "__main__":
    main()
