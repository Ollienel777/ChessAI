import modal

app = modal.App("lichess-pipeline")

# Shared volume for positions and labeled positions
volume = modal.Volume.from_name("lichess-data", create_if_missing=True)

# --------- IMAGE FOR POSITION GENERATION (Lichess DB) ---------
image_gen = (
    modal.Image.debian_slim()
    .pip_install("python-chess", "zstandard", "requests")
)

@app.function(image=image_gen, volumes={"/data": volume}, timeout=60 * 60)
def generate_positions(
    url: str,
    max_positions: int = 1_000_000,
    min_elo: int = 1500,
):
    """
    Stream a lichess .pgn.zst file from `url`, extract up to `max_positions`
    midgame positions into /data/positions.csv (in the Modal volume).
    """
    import io
    import csv
    import random
    import requests
    import zstandard as zstd
    import chess.pgn
    from pathlib import Path

    out_dir = Path("/data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "positions.csv"

    f_out = out_path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(f_out)
    writer.writerow(["fen"])

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    dctx = zstd.ZstdDecompressor()
    stream_reader = dctx.stream_reader(resp.raw)
    text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")

    num_positions = 0
    game_counter = 0

    while num_positions < max_positions:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break  # end of file

        game_counter += 1

        # Filter by player Elo if headers present
        try:
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
        except ValueError:
            continue

        if white_elo < min_elo or black_elo < min_elo:
            continue

        moves = list(game.mainline_moves())
        if len(moves) < 10:
            continue

        # midgame corridor in plies (e.g. moves 10–60)
        ply_start = min(20, len(moves))
        ply_end = min(120, len(moves))
        if ply_start >= ply_end:
            continue

        num_samples_for_game = 2
        for _ in range(num_samples_for_game):
            ply_index = random.randint(ply_start, ply_end - 1)
            board = game.board()
            for m in moves[:ply_index]:
                board.push(m)

            writer.writerow([board.fen()])
            num_positions += 1
            if num_positions >= max_positions:
                break

        if game_counter % 1000 == 0:
            print(f"Processed {game_counter} games, collected {num_positions} positions")

    f_out.close()
    print(f"Done. Wrote {num_positions} positions to {out_path}")


# --------- IMAGE FOR STOCKFISH LABELING ---------
image_sf = (
    modal.Image.debian_slim()
    .apt_install("stockfish")            # installs the engine binary
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
        raise FileNotFoundError(
            f"{positions_path} not found. Did you run generate_positions into the same volume?"
        )

    engine_path = "/usr/games/stockfish"
    engine = chess.engine.SimpleEngine.popen_uci([engine_path])
    print("Stockfish engine started.")

    def score_to_value(score: chess.engine.PovScore) -> float:
        s = score.pov(chess.WHITE)
        if s.is_mate():
            return 1.0 if s.mate() > 0 else -1.0
        cp = s.score() or 0
        pawns = max(-3.0, min(3.0, cp / 100.0))
        return pawns / 3.0  # normalize ~[-1, 1]

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
