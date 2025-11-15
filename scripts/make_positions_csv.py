# scripts/make_positions_csv.py
import csv
from pathlib import Path
import argparse
import io

import chess
import chess.pgn


def result_to_target(result_str: str) -> float:
    if result_str == "1-0":
        return 1.0
    if result_str == "0-1":
        return -1.0
    return 0.0  # draws and others


def extract_positions(pgn_path: Path,
                      csv_path: Path,
                      sample_every: int = 4):
    """
    Read PGNs, write a CSV with columns fen,result
    sampling every N plies to keep dataset manageable.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    num_games = 0
    num_positions = 0

    with pgn_path.open("r", encoding="utf-8") as f_in, \
            csv_path.open("w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(["fen", "result"])

        while True:
            game = chess.pgn.read_game(f_in)
            if game is None:
                break

            num_games += 1
            result = game.headers.get("Result", "1/2-1/2")
            target = result_to_target(result)

            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                board.push(move)
                ply += 1
                if ply % sample_every == 0:
                    writer.writerow([board.fen(), target])
                    num_positions += 1

    print(f"Extracted {num_positions} positions from {num_games} games into {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=str, default="data/high_elo_games.pgn")
    parser.add_argument("--out", type=str, default="data/positions.csv")
    parser.add_argument("--sample-every", type=int, default=4)
    args = parser.parse_args()

    extract_positions(Path(args.pgn), Path(args.out), sample_every=args.sample_every)
