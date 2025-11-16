# chess-value-trainer/modal_train.py

import modal

# Name your app
app = modal.App("chess-value-trainer")

# Define the environment image (Python + requirements.txt)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "python-chess", "numpy")
)

# If you want to run on GPU, set gpu="any" or a specific type like "T4"
# For CPU only, remove gpu=...
@app.function(image=image, gpu="any", timeout=60 * 60)
def run_training(epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
    """
    This function runs inside Modal's cloud.
    It imports your train.py and calls train().
    """
    import os
    from pathlib import Path

    # Ensure we're in the project root inside the container
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    # Now we can import from src/ because it's in the same directory
    from src.train import train

    print("Starting training on Modal...")
    print(f"epochs={epochs}, batch_size={batch_size}, lr={lr}")

    train(epochs=epochs, batch_size=batch_size, lr=lr)

    # Model will be saved to src/valuenet.pt
    out_path = project_root / "src" / "valuenet.pt"
    if out_path.exists():
        print(f"Training complete. Saved model to {out_path}")
    else:
        print("WARNING: Training finished but valuenet.pt was not found.")
