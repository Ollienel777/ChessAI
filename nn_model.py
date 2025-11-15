# nn_model.py
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encode board as (13, 8, 8) tensor:
    12 piece planes (6 white + 6 black) + 1 side-to-move plane.
    """
    planes = torch.zeros(13, 8, 8, dtype=torch.float32)

    # piece planes
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type   # 1..6
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane_idx = color_offset + (piece_type - 1)

        rank = chess.square_rank(square)      # 0 (a1) to 7 (a8)
        file = chess.square_file(square)      # 0..7
        row = 7 - rank                        # so White is at bottom
        col = file
        planes[plane_idx, row, col] = 1.0

    # side-to-move plane: +1 for white to move, -1 for black
    stm_value = 1.0 if board.turn == chess.WHITE else -1.0
    planes[12, :, :] = stm_value

    return planes


class ValueNet(nn.Module):
    """
    Small CNN that outputs a single scalar value:
    positive = good for White, negative = good for Black.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, 13, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # (batch, 1)
        return x
