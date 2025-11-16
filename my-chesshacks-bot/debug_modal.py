import modal
import os

stub = modal.App("debug-fs")
@stub.local_entrypoint()
def main():
    print("This runs locally (optional)")

@stub.function()
def list_dir(path="/"):
    print("Listing:", path)
    for root, dirs, files in os.walk(path):
        print(root)
        for f in files:
            print("  -", f)
        for d in dirs:
            print("  /", d)