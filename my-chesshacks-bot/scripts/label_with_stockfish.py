import csv
from pathlib import Path

import chess
import chess.engine


ROOT = Path(__file__).resolve().parents[1]  # .../my-chesshacks-bot
DATA_CSV = ROOT / "data" / "positions.csv"          # input (old labels)
OUT_CSV  = ROOT / "data" / "sf_positions.csv"       # output (Stockfish labels)

# CHANGE THIS to where your Stockfish binary is:
STOCKFISH_PATH = ROOT / "stockfish.exe"   # e.g. put stockfish.exe in my-chesshacks-bot/


def cp_to_target(cp: int) -> float:
    """
    Convert centipawn score to [-1,1] roughly.
    +400 cp ~ +1, -400 cp ~ -1, clamp outer values.
    """
    x = cp / 400.0
    if x > 1.0:
        x = 1.0
    if x < -1.0:
        x = -1.0
    return float(x)


def main(max_positions: int | None = 300000, depth: int = 8):
    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))

    with DATA_CSV.open("r", encoding="utf-8") as f_in, \
         OUT_CSV.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["fen", "sf_score"])

        for i, row in enumerate(reader):
            if max_positions is not None and i >= max_positions:
                break

            fen = row["fen"]
            board = chess.Board(fen)

            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].pov(chess.WHITE)

            # score can be mate or cp; convert both to a numeric cp value
            if score.is_mate():
                # big positive for mate for white, big negative for mate for black
                cp = 10000 if score.mate() > 0 else -10000
            else:
                cp = score.score()  # centipawns

            target = cp_to_target(cp)
            writer.writerow([fen, target])

            if (i + 1) % 200 == 0:
                print(f"Labeled {i+1} positions...")

    engine.quit()
    print(f"Done. Wrote Stockfish labels to {OUT_CSV}")


if __name__ == "__main__":
    main()
