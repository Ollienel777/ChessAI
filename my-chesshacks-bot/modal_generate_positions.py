import modal

app = modal.App("chess-training")

# Shared volume for positions / labels / model
volume = modal.Volume.from_name("lichess-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "python-chess", "numpy")
)

@app.function(
    image=image,
    volumes={"/modal_data": volume},  # mount Modal volume
    mounts=[                         # mount your *local* src/ into the container
        modal.Mount.from_local_dir("src", remote_path="/root/src")
    ],
    timeout=60 * 105,
)
def train_remote(epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
    """
    Run src/train.py inside Modal, using sf_positions.csv from the lichess-data volume.
    Saves valuenet.pt back into the volume.
    """
    import shutil
    from pathlib import Path
    import sys

    # We KNOW src is at /root/src inside the container
    project_root = Path("/root")
    src_dir = project_root / "src"
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    # Where the Modal volume is mounted
    volume_dir = Path("/modal_data")

    # 1) Copy sf_positions.csv from volume into project data dir
    sf_src = volume_dir / "sf_positions.csv"
    if not sf_src.exists():
        raise FileNotFoundError(f"{sf_src} not found in Modal volume 'lichess-data'")

    sf_dst = data_dir / "sf_positions.csv"
    print(f"[INFO] Copying {sf_src} -> {sf_dst}")
    shutil.copy(sf_src, sf_dst)

    # 2) Import your local train() and run it
    sys.path.insert(0, str(project_root))  # lets us import src.*

    print(f"[INFO] sys.path[0]: {sys.path[0]}")
    from src.train import train as local_train

    print(f"[INFO] Starting training: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    local_train(epochs=epochs, batch_size=batch_size, lr=lr)

    # 3) Copy resulting valuenet.pt back into the Modal volume
    model_path = src_dir / "valuenet.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found after training")

    model_dst = volume_dir / "valuenet.pt"
    print(f"[INFO] Copying {model_path} -> {model_dst}")
    shutil.copy(model_path, model_dst)

    print("[INFO] Training complete and model saved to volume.")
