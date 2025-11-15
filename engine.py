# engine.py
import math
import chess
import torch

from nn_model import board_to_tensor, ValueNet

MATE_SCORE = 10_000.0


class NNChessEngine:
    def __init__(self, model: ValueNet, device: str = "cpu", max_depth: int = 3):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.max_depth = max_depth

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> float:
        """
        Pure neural evaluation.
        If you replaced this with `return 0.0`, your engine would basically
        play random moves – so it clearly depends on the network.
        """
        if board.is_game_over():
            if board.is_checkmate():
                # side to move is checkmated
                return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
            return 0.0  # draw / stalemate / repetition

        x = board_to_tensor(board).unsqueeze(0).to(self.device)  # (1, 13, 8, 8)
        value = self.model(x).item()
        return value  # + good for white, - good for black

    def _ordered_moves(self, board: chess.Board):
        # Simple move ordering: captures first.
        moves = list(board.legal_moves)

        def score(move: chess.Move) -> int:
            return 1 if board.is_capture(move) else 0

        return sorted(moves, key=score, reverse=True)

    def _alphabeta(self, board: chess.Board, depth: int,
                   alpha: float, beta: float, maximizing: bool) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        if maximizing:
            value = -math.inf
            for move in self._ordered_moves(board):
                board.push(move)
                value = max(value, self._alphabeta(board, depth - 1,
                                                   alpha, beta, False))
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for move in self._ordered_moves(board):
                board.push(move)
                value = min(value, self._alphabeta(board, depth - 1,
                                                   alpha, beta, True))
                board.pop()
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def select_move(self, board: chess.Board) -> chess.Move:
        best_move = None
        maximizing = (board.turn == chess.WHITE)
        best_score = -math.inf if maximizing else math.inf

        for move in self._ordered_moves(board):
            board.push(move)
            score = self._alphabeta(board,
                                    self.max_depth - 1,
                                    -math.inf, math.inf,
                                    not maximizing)
            board.pop()

            if maximizing and (score > best_score or best_move is None):
                best_score = score
                best_move = move
            if (not maximizing) and (score < best_score or best_move is None):
                best_score = score
                best_move = move

        return best_move
