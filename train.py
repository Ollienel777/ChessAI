# train.py
import csv
from pathlib import Path

import chess
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from nn_model import board_to_tensor, ValueNet


class PositionDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.samples = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["fen"], float(row["result"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, result = self.samples[idx]
        board = chess.Board(fen)
        x = board_to_tensor(board)            # (13, 8, 8)
        y = torch.tensor([result], dtype=torch.float32)
        return x, y


def train(csv_path="data/positions.csv",
          epochs: int = 3,
          batch_size: int = 64,
          lr: float = 1e-3,
          out_path: str = "valuenet.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PositionDataset(Path(csv_path))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ValueNet().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).squeeze(1)  # shape (batch,)

            opt.zero_grad()
            preds = model(x).squeeze(1)  # (batch,)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    train()
