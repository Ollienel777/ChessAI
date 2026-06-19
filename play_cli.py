# play_cli.py
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "my-chesshacks-bot" / "src"
sys.path.insert(0, str(SRC_DIR))

import chess
import torch

from nn_model import ValueNet, board_to_tensor

WEIGHTS_PATH = SRC_DIR / "valuenet.pt"
INF = 1e9


class NNChessEngine:
    def __init__(self, model, device, max_depth=2):
        self.model = model
        self.device = device
        self.max_depth = max_depth

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> float:
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        return torch.tanh(self.model(x)).item()

    def eval_terminal(self, board: chess.Board) -> float:
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return 1.0
            if result == "0-1":
                return -1.0
            return 0.0
        return self.evaluate(board)

    def alpha_beta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or board.is_game_over():
            return self.eval_terminal(board)

        if board.turn == chess.WHITE:
            value = -INF
            for move in board.legal_moves:
                board.push(move)
                value = max(value, self.alpha_beta(board, depth - 1, alpha, beta))
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = INF
            for move in board.legal_moves:
                board.push(move)
                value = min(value, self.alpha_beta(board, depth - 1, alpha, beta))
                board.pop()
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    def select_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        white_to_move = board.turn == chess.WHITE
        best_score = -INF if white_to_move else INF
        best_move = legal_moves[0]
        alpha, beta = -INF, INF

        for move in legal_moves:
            board.push(move)
            score = self.alpha_beta(board, self.max_depth - 1, alpha, beta)
            board.pop()

            if white_to_move and score > best_score:
                best_score = score
                best_move = move
                alpha = max(alpha, best_score)
            if not white_to_move and score < best_score:
                best_score = score
                best_move = move
                beta = min(beta, best_score)

        return best_move


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ValueNet().to(device)

    try:
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model from {WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"WARNING: {WEIGHTS_PATH} not found, using random weights (engine will suck).")
    model.eval()

    engine = NNChessEngine(model=model, device=device, max_depth=2)
    board = chess.Board()

    print("You are WHITE. Enter moves in UCI (e2e4, g1f3, etc.). Type 'quit' to exit.\n")
    print(board, "\n")

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            mv = input("Your move: ").strip()
            if mv.lower() in ("quit", "exit"):
                break
            try:
                move = chess.Move.from_uci(mv)
            except ValueError:
                print("Bad format.")
                continue
            if move not in board.legal_moves:
                print("Illegal move.")
                continue
            board.push(move)
        else:
            print("Engine thinking...")
            move = engine.select_move(board)
            print("Engine plays:", move.uci())
            board.push(move)

        print(board, "\n")

    print("Game over:", board.result())


if __name__ == "__main__":
    main()
