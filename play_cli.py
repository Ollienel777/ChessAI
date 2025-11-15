# play_cli.py
import chess
import torch

from nn_model import ValueNet
from engine import NNChessEngine


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ValueNet()

    # Load your trained weights (created in step 7).
    try:
        state = torch.load("valuenet.pt", map_location=device)
        model.load_state_dict(state)
        print("Loaded model from valuenet.pt")
    except FileNotFoundError:
        print("WARNING: valuenet.pt not found, using random weights (engine will suck).")

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
