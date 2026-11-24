import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from .dataset import build_datasets
from .model import RoadSignOcclusionCNN

def run_training(args):
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds = build_datasets(args.data_dir, args.limit_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    model = RoadSignOcclusionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, sign_labels, occ_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = images.to(device)
            sign_labels = sign_labels.to(device)
            occ_labels = occ_labels.to(device)
            optimizer.zero_grad()
            sign_logits, occ_logits = model(images)
            loss = criterion(sign_logits, sign_labels) + args.occlusion_weight * criterion(occ_logits, occ_labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for images, sign_labels, occ_labels in val_loader:
                images, sign_labels, occ_labels = images.to(device), sign_labels.to(device), occ_labels.to(device)
                sign_logits, occ_logits = model(images)
                loss = criterion(sign_logits, sign_labels) + args.occlusion_weight * criterion(occ_logits, occ_labels)
                val_running += loss.item() * images.size(0)
        val_loss = val_running / len(val_ds)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}: train loss={train_loss:.4f}, validation loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output / "best_model.pth")

    with open(output / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Combined loss")
    plt.title("Road Sign CNN Learning Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "learning_curves.png", dpi=160)
    print(f"Best model saved to {output / 'best_model.pth'}")
    return test_ds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--occlusion-weight", type=float, default=1.0)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    run_training(parse_args())
