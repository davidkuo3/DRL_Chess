import chess
import torch
from drl_chess import ChessEnv, DRLChessNet, ActionSelector

def play_game():
    print("Welcome to DRL_Chess!")
    print("You are playing as White. Enter your moves in UCI format (e.g., e2e4, g1f3).")
    print("Type 'quit' to exit the game.\n")

    env = ChessEnv()
    device = torch.device("cpu")
    
    # Initialize the neural network (currently untrained)
    net = DRLChessNet().to(device)
    net.eval()
    
    # Action selector for the AI (using epsilon-greedy for now)
    # We set epsilon to 0.1 so it mostly uses the network (even if untrained), but sometimes explores.
    # If you want it to be completely random, set epsilon=1.0.
    selector = ActionSelector(method="epsilon-greedy", epsilon=0.1)

    while not env.board.is_game_over():
        print("\n" + "="*30)
        print("Current Board:")
        print(env.board)
        print("="*30 + "\n")

        if env.board.turn == chess.WHITE:
            # User's turn
            user_move = input("Your move (UCI format): ").strip()
            
            if user_move.lower() == 'quit':
                print("Game aborted by user.")
                break
                
            try:
                # Check if the move is legal and push it
                if chess.Move.from_uci(user_move) in env.board.legal_moves:
                    env.step(user_move)
                else:
                    print("Illegal move! Please try again.")
            except ValueError:
                print("Invalid format! Please use UCI format like 'e2e4'.")
                
        else:
            # AI's turn
            print("AI is thinking...")
            ai_move = selector.select_action(env, net, device)
            print(f"AI played: {ai_move}")
            env.step(ai_move)

    if env.board.is_game_over():
        print("\n" + "="*30)
        print("Final Board:")
        print(env.board)
        print("="*30 + "\n")
        
        result = env.board.result()
        print(f"Game Over! Result: {result}")
        if result == '1-0':
            print("Congratulations! You won!")
        elif result == '0-1':
            print("AI won. Better luck next time!")
        else:
            print("It's a draw!")

if __name__ == "__main__":
    play_game()
