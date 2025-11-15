from .utils import chess_manager, GameContext
import chess
import torch
import time
import math
from pathlib import Path

from .nn_model import ValueNet, board_to_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = Path(__file__).with_name("valuenet.pt")

# ------------------------
# Load neural network
# ------------------------

try:
    _model = ValueNet().to(DEVICE)
    _model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    _model.eval()
    print(f"[INFO] Loaded neural network weights from {WEIGHTS_PATH}")
except Exception as e:
    print(f"[WARNING] Could not load weights ({e}) — using RANDOM model (will play badly)")
    _model = ValueNet().to(DEVICE)
    _model.eval()


@torch.no_grad()
def evaluate(board: chess.Board) -> float:
    """
    Pure neural evaluation.
    Output squashed to [-1, 1].
    Positive = good for White, negative = good for Black.
    """
    x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    v = _model(x)          # (1,1)
    return torch.tanh(v).item()


INF = 1e9

# ------------------------
# Search
# ------------------------

def eval_terminal(board: chess.Board) -> float:
    """
    Exact outcome for terminal positions, otherwise NN.
    Always from White's perspective.
    """
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        return 0.0
    return evaluate(board)


def alpha_beta(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """
    Standard alpha-beta search using the NN as leaf evaluator.
    Returns value from White's perspective.
    """
    if depth == 0 or board.is_game_over():
        return eval_terminal(board)

    if board.turn == chess.WHITE:
        value = -INF
        for move in board.legal_moves:
            board.push(move)
            value = max(value, alpha_beta(board, depth - 1, alpha, beta))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = INF
        for move in board.legal_moves:
            board.push(move)
            value = min(value, alpha_beta(board, depth - 1, alpha, beta))
            board.pop()
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def search_root(board: chess.Board, depth: int):
    """
    Search all legal moves to given depth and pick the best one.
    Returns (best_move, scores_in_move_order).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, []

    scores = []
    white_to_move = board.turn == chess.WHITE
    best_score = -INF if white_to_move else INF
    best_move = legal_moves[0]

    for move in legal_moves:
        board.push(move)
        score = alpha_beta(board, depth - 1, -INF, INF)
        board.pop()
        scores.append(score)

        if white_to_move and score > best_score:
            best_score = score
            best_move = move
        if not white_to_move and score < best_score:
            best_score = score
            best_move = move

    return best_move, scores

# ------------------------
# Entry / reset hooks
# ------------------------

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Called every time a move is needed.

    We run a small alpha-beta search using ONLY the neural net
    as the evaluation function, then log move probabilities.
    """
    # This is the "true" game board from the framework
    live_board = ctx.board

    if live_board.is_game_over():
        ctx.logProbabilities({})
        raise ValueError("Game over")

    # Work on a COPY so we don't mutate ctx.board during search
    board_for_search = live_board.copy()

    # Adjust if too slow / too weak
    SEARCH_DEPTH = 3

    # Run search on the copy
    best_move, scores = search_root(board_for_search, depth=SEARCH_DEPTH)

    # Legal moves for the REAL board
    legal_moves = list(live_board.legal_moves)

    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Safety check: ensure chosen move is legal in the live position
    if best_move not in legal_moves:
        # Fallback: pick the legal move with the best score if we can match it,
        # otherwise just take the first legal move.
        try:
            # Map scores back to moves from the search root (board_for_search)
            # and filter only those that are still legal in live_board
            move_to_score = {}
            for m, s in zip(board_for_search.legal_moves, scores):
                move_to_score[m] = s

            white_to_move = live_board.turn == chess.WHITE
            # Filter moves that are actually legal now
            candidate_moves = [m for m in legal_moves if m in move_to_score]

            if candidate_moves:
                if white_to_move:
                    best_move = max(candidate_moves, key=lambda m: move_to_score[m])
                else:
                    best_move = min(candidate_moves, key=lambda m: move_to_score[m])
            else:
                # As a last resort, just pick the first legal move
                best_move = legal_moves[0]
        except Exception:
            best_move = legal_moves[0]

    # Rebuild scores in the order of live_board's legal moves for logging
    # If we can't align them perfectly, just assign uniform probabilities.
    probs_dict = {}
    try:
        # We'll attempt to map scores by recomputing at depth 1 (cheap)
        tmp_scores = []
        for move in legal_moves:
            tmp_board = live_board.copy()
            tmp_board.push(move)
            tmp_scores.append(alpha_beta(tmp_board, SEARCH_DEPTH - 1, -INF, INF))

        white_to_move = live_board.turn == chess.WHITE
        adj = tmp_scores if white_to_move else [-s for s in tmp_scores]

        max_s = max(adj)
        exps = [math.exp(s - max_s) for s in adj]
        total = sum(exps) if exps else 0.0

        if total <= 0:
            probs = [1.0 / len(legal_moves)] * len(legal_moves)
        else:
            probs = [e / total for e in exps]

        probs_dict = {m: p for m, p in zip(legal_moves, probs)}
    except Exception:
        # If anything goes wrong computing probs, default to uniform
        uniform_p = 1.0 / len(legal_moves)
        probs_dict = {m: uniform_p for m in legal_moves}

    ctx.logProbabilities(probs_dict)

    print(f"Depth {SEARCH_DEPTH}, chosen {best_move}")
    time.sleep(0.02)

    return best_move