import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from .model import RoadSignOcclusionCNN

def evaluate(test_ds, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadSignOcclusionCNN().to(device)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.eval()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    sign_true, sign_pred, occ_true, occ_pred = [], [], [], []
    with torch.no_grad():
        for images, sign_labels, occ_labels in loader:
            sign_logits, occ_logits = model(images.to(device))
            sign_pred.extend(sign_logits.argmax(1).cpu().tolist())
            occ_pred.extend(occ_logits.argmax(1).cpu().tolist())
            sign_true.extend(sign_labels.tolist())
            occ_true.extend(occ_labels.tolist())

    metrics = {
        "sign_classification_accuracy": accuracy_score(sign_true, sign_pred),
        "occlusion_accuracy": accuracy_score(occ_true, occ_pred),
        "occlusion_precision": precision_score(occ_true, occ_pred, zero_division=0),
        "occlusion_recall": recall_score(occ_true, occ_pred, zero_division=0),
        "occlusion_f1": f1_score(occ_true, occ_pred, zero_division=0),
        "occlusion_confusion_matrix": confusion_matrix(occ_true, occ_pred).tolist()
    }
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/evaluation_report.json", "w") as f:
        json.dump(metrics, f, indent=2)
    cm = confusion_matrix(occ_true, occ_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Occlusion Detection Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.xticks([0, 1], ["Clear", "Occluded"])
    plt.yticks([0, 1], ["Clear", "Occluded"])
    for (i, j), value in __import__("numpy").ndenumerate(cm):
        plt.text(j, i, str(value), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("outputs/occlusion_confusion_matrix.png", dpi=160)
    print("\nTEST RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("Saved outputs/evaluation_report.json")
    return metrics
