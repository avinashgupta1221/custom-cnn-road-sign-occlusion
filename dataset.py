import random
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

IMAGE_SIZE = 64

class SyntheticOcclusion:
    """Draws random natural/paint-like patches over a traffic-sign image."""
    def __call__(self, image):
        image = image.copy().convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        w, h = image.size
        count = random.randint(1, 4)
        palettes = [(35, 75, 28, 220), (92, 58, 35, 225),
                    (25, 25, 25, 220), (130, 120, 95, 210)]
        for _ in range(count):
            color = random.choice(palettes)
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            rw, rh = random.randint(w//8, w//2), random.randint(h//10, h//2)
            x2, y2 = min(w, x1+rw), min(h, y1+rh)
            shape = random.choice(["ellipse", "polygon", "line"])
            if shape == "ellipse":
                draw.ellipse((x1, y1, x2, y2), fill=color)
            elif shape == "polygon":
                pts = [(random.randint(x1, max(x1, x2)),
                        random.randint(y1, max(y1, y2))) for _ in range(6)]
                draw.polygon(pts, fill=color)
            else:
                draw.line((x1, y1, x2, y2), fill=color, width=random.randint(3, 9))
        return image.filter(ImageFilter.GaussianBlur(radius=0.25))

class GTSRBOcclusionDataset(Dataset):
    def __init__(self, base_dataset, training=False, occlusion_probability=0.5):
        self.base = base_dataset
        self.training = training
        self.occlusion_probability = occlusion_probability
        self.occluder = SyntheticOcclusion()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        is_occluded = int(random.random() < self.occlusion_probability)
        if is_occluded:
            image = self.occluder(image)
        return self.transform(image), int(label), is_occluded

def build_datasets(root="./data", limit_train=None):
    train_base = datasets.GTSRB(root=root, split="train", download=True)
    test_base = datasets.GTSRB(root=root, split="test", download=True)
    if limit_train:
        indices = list(range(min(limit_train, len(train_base))))
        train_base = torch.utils.data.Subset(train_base, indices)

    val_size = max(1, int(0.15 * len(train_base)))
    train_size = len(train_base) - val_size
    train_part, val_part = random_split(
        train_base, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_ds = GTSRBOcclusionDataset(train_part, training=True, occlusion_probability=0.5)
    val_ds = GTSRBOcclusionDataset(val_part, occlusion_probability=0.5)
    test_ds = GTSRBOcclusionDataset(test_base, occlusion_probability=0.5)
    return train_ds, val_ds, test_ds
