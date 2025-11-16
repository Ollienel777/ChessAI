import modal

app = modal.App("chess-training")

volume = modal.Volume.from_name("lichess-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "python-chess", "numpy")
    .add_local_dir(".", remote_path="/project")
)

@app.function(image=image, volumes={"/modal_data": volume}, timeout=60 * 600)
def train_remote(epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
    """
    Run src/train.py inside Modal, using sf_positions.csv from the lichess-data volume.
    Saves/updates /modal_data/valuenet.pt every epoch.
    """
    import shutil
    from pathlib import Path
    import sys

    project_root = Path("/project")
    src_dir = project_root / "src"
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    volume_dir = Path("/modal_data")

    # 1) Copy sf_positions.csv from volume into project data dir
    sf_src = volume_dir / "sf_positions.csv"
    if not sf_src.exists():
        raise FileNotFoundError(f"{sf_src} not found in Modal volume 'lichess-data'")

    sf_dst = data_dir / "sf_positions.csv"
    print(f"[INFO] Copying {sf_src} -> {sf_dst}")
    shutil.copy(sf_src, sf_dst)

    # 2) Import your local train() and run it, writing checkpoints directly to the volume
    sys.path.append(str(src_dir))
    from train import train as local_train

    out_path = volume_dir / "valuenet.pt"
    print(f"[INFO] Starting training to {out_path}: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    local_train(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        csv_path=sf_dst,
        out_path=out_path,
        checkpoint_every=1,  # ✅ save each epoch
    )

    print("[INFO] Training complete; latest weights are in the volume.")
