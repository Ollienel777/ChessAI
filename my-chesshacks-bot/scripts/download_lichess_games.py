import argparse
import json
from pathlib import Path

import requests


def download_games(username: str,
                   out_path: Path,
                   max_games: int = 1000,
                   min_elo: int = 2300):
    url = f"https://lichess.org/api/games/user/{username}"
    headers = {"Accept": "application/x-ndjson"}
    params = {
        "max": max_games,
        "rated": "true",
        "analysed": "false",
        "pgnInJson": "true",
        "moves": "true",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0

    with requests.get(url, headers=headers, params=params, stream=True) as r, \
            out_path.open("w", encoding="utf-8") as f_out:

        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            total += 1
            game = json.loads(line)
            white = game["players"].get("white", {})
            black = game["players"].get("black", {})
            w_elo = white.get("rating", 0)
            b_elo = black.get("rating", 0)

            if w_elo < min_elo or b_elo < min_elo:
                continue

            pgn = game.get("pgn")
            if not pgn:
                continue

            f_out.write(pgn)
            f_out.write("\n\n")
            kept += 1

    print(f"Processed {total} games, kept {kept} high-ELO games to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("username", help="Lichess username")
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--min-elo", type=int, default=2300)
    parser.add_argument("--out", type=str, default="data/high_elo_games.pgn")
    args = parser.parse_args()

    download_games(args.username, Path(args.out),
                   max_games=args.max_games, min_elo=args.min_elo)
