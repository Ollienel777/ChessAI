import modal

app = modal.App("lichess-labeler")

# Shared volume for positions and labeled positions
volume = modal.Volume.from_name("lichess-data", create_if_missing=True)

# Image: install system stockfish + python-chess
image_sf = (
    modal.Image.debian_slim()
    .apt_install("stockfish")          # installs the engine binary
    .pip_install("python-chess")
)

@app.function(image=image_sf, volumes={"/data": volume}, timeout=60 * 60)
def label_with_stockfish(max_positions: int = 1_000_000, depth: int = 12):
    """
    Read /data/positions.csv, evaluate with Stockfish, and write /data/sf_positions.csv.
    """
    import csv
    from pathlib import Path
    import chess
    import chess.engine

    positions_path = Path("/data") / "positions.csv"
    out_path = Path("/data") / "sf_positions.csv"

    if not positions_path.exists():
        raise FileNotFoundError(f"{positions_path} not found. Did you run generate_positions into the same volume?")

    # Path where Debian's stockfish binary lives
    engine_path = "/usr/games/stockfish"

    engine = chess.engine.SimpleEngine.popen_uci([engine_path])
    print("Stockfish engine started.")

    def score_to_value(score: chess.engine.PovScore) -> float:
        s = score.pov(chess.WHITE)
        if s.is_mate():
            return 1.0 if s.mate() > 0 else -1.0
        cp = s.score() or 0
        # clamp to [-3,3] pawns and scale to [-1,1]
        pawns = max(-3.0, min(3.0, cp / 100.0))
        return pawns / 3.0

    with positions_path.open("r", encoding="utf-8") as f_in, \
         out_path.open("w", encoding="utf-8", newline="") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["fen", "sf_score"])
        writer.writeheader()

        count = 0
        for row in reader:
            fen = row["fen"]
            board = chess.Board(fen)

            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            val = score_to_value(info["score"])

            writer.writerow({"fen": fen, "sf_score": val})
            count += 1

            if count % 1000 == 0:
                print(f"Labeled {count} positions")

            if count >= max_positions:
                break

    engine.quit()
    print(f"Done. Wrote {count} labeled positions to {out_path}")
