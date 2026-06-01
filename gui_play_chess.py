import tkinter as tk
from tkinter import messagebox, simpledialog
import chess
import torch
import threading
from drl_chess import ChessEnv, DRLChessNet, ActionSelector

# Unicode mapping for chess pieces
PIECE_UNICODE = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DRL_Chess GUI - Play with Mouse")
        
        self.square_size = 80
        self.canvas = tk.Canvas(root, width=8*self.square_size, height=8*self.square_size)
        self.canvas.pack()

        # Initialize Environment & AI
        import os
        self.env = ChessEnv()
        self.device = torch.device("cpu")
        self.net = DRLChessNet().to(self.device)
        
        if os.path.exists("model.pth"):
            self.net.load_state_dict(torch.load("model.pth", map_location=self.device, weights_only=True))
            print("Loaded trained model from model.pth")
            
        self.net.eval()
        # Set epsilon=0.0 so the AI strictly uses the trained policy
        self.selector = ActionSelector(method="epsilon-greedy", epsilon=0.0)

        self.selected_square = None
        self.ai_thinking = False

        self.draw_board()
        self.canvas.bind("<Button-1>", self.on_click)

    def draw_board(self):
        self.canvas.delete("all")
        colors = ["#F0D9B5", "#B58863"] # Light and Dark square colors

        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                
                # Highlight selected square
                if self.selected_square == (row, col):
                    color = "#baca44"

                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                # chess.Board uses 0 for a1 (bottom-left) to 63 for h8 (top-right)
                # Tkinter uses (0,0) for top-left. So we must invert the row.
                square_idx = chess.square(col, 7 - row)
                piece = self.env.board.piece_at(square_idx)
                
                if piece:
                    piece_char = PIECE_UNICODE[piece.symbol()]
                    piece_color = "black" if piece.color == chess.BLACK else "white"
                    # Add subtle outline for white pieces to make them visible on light squares
                    if piece.color == chess.WHITE:
                        self.canvas.create_text(x1 + self.square_size/2, y1 + self.square_size/2, 
                                                text=piece_char, font=("Segoe UI Symbol", 48), fill="black")
                        self.canvas.create_text(x1 + self.square_size/2, y1 + self.square_size/2, 
                                                text=piece_char, font=("Segoe UI Symbol", 47), fill="white")
                    else:
                        self.canvas.create_text(x1 + self.square_size/2, y1 + self.square_size/2, 
                                                text=piece_char, font=("Segoe UI Symbol", 48), fill="black")

    def on_click(self, event):
        if self.ai_thinking or self.env.board.is_game_over():
            return

        col = event.x // self.square_size
        row = event.y // self.square_size

        if self.selected_square:
            start_col = self.selected_square[1]
            start_row = 7 - self.selected_square[0]
            end_col = col
            end_row = 7 - row
            
            start_sq_name = chess.square_name(chess.square(start_col, start_row))
            end_sq_name = chess.square_name(chess.square(end_col, end_row))
            move_uci = start_sq_name + end_sq_name

            # Check for pawn promotion
            move_obj = chess.Move.from_uci(move_uci)
            if move_obj not in self.env.board.legal_moves:
                if chess.Move.from_uci(move_uci + 'q') in self.env.board.legal_moves:
                    promo = simpledialog.askstring("Pawn Promotion", "Promote to (q=Queen, r=Rook, n=Knight, b=Bishop):", parent=self.root)
                    if promo and promo.lower() in ['q', 'r', 'n', 'b']:
                        move_uci += promo.lower()
                    else:
                        move_uci += 'q'  # Default to queen
                    move_obj = chess.Move.from_uci(move_uci)

            if move_obj in self.env.board.legal_moves:
                self.env.step(move_uci)
                self.selected_square = None
                self.draw_board()
                self.check_game_over()
                
                # Trigger AI turn if game is not over
                if not self.env.board.is_game_over():
                    self.ai_thinking = True
                    self.root.after(100, self.ai_turn)
            else:
                # If illegal move, just deselect or select the new piece if it's user's color
                sq_idx = chess.square(col, 7 - row)
                piece = self.env.board.piece_at(sq_idx)
                if piece and piece.color == self.env.board.turn:
                    self.selected_square = (row, col)
                else:
                    self.selected_square = None
        else:
            sq_idx = chess.square(col, 7 - row)
            piece = self.env.board.piece_at(sq_idx)
            if piece and piece.color == self.env.board.turn:
                self.selected_square = (row, col)

        self.draw_board()

    def ai_turn(self):
        ai_move, _ = self.selector.select_action(self.env, self.net, self.device)
        if ai_move:
            self.env.step(ai_move)
        
        self.ai_thinking = False
        self.draw_board()
        self.check_game_over()

    def check_game_over(self):
        if self.env.board.is_game_over():
            result = self.env.board.result()
            if result == '1-0':
                msg = "White (You) Won!"
            elif result == '0-1':
                msg = "Black (AI) Won!"
            else:
                msg = "It's a Draw!"
            messagebox.showinfo("Game Over", f"{msg}\nResult: {result}")

if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    gui = ChessGUI(root)
    root.mainloop()
