# Construct a Custom CNN to Detect Road Sign Occlusion

A multi-task CNN that predicts both the traffic-sign class and whether the sign is occluded. The project uses the public **GTSRB** traffic-sign dataset and creates synthetic occlusions during training and evaluation.

## Features
- Downloads GTSRB automatically through `torchvision.datasets.GTSRB`
- Creates synthetic leaf/mud/graffiti-like occluders using random polygons, blobs, and strokes
- Custom CNN with two output heads: 43-class sign classification and binary occlusion detection
- Train/validation/test split
- Accuracy, precision, recall, F1-score, confusion matrix, and learning curves
- Saves the trained model and evaluation report

## Project structure
```text
custom_cnn_road_sign_occlusion/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── outputs/
├── requirements.txt
├── run.py
└── README.md
```

## Setup
Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the project
```bash
python run.py
```
The first run downloads GTSRB (internet required). Training uses synthetic occlusion augmentation. Outputs are written to `outputs/`.

Optional smaller/faster run:
```bash
python run.py --epochs 3 --batch-size 64 --limit-train 5000
```

## Model and labels
The CNN has a shared convolutional feature extractor and two task-specific heads:
1. **Sign head:** predicts one of 43 GTSRB traffic-sign classes.
2. **Occlusion head:** predicts clear (0) or occluded (1).

The total loss is the sum of cross-entropy losses for both tasks. Training images are randomly occluded; validation and test data include a controlled mixture of clear and occluded examples.

## Notes
- GTSRB is a German traffic-sign benchmark, not a complete representation of every country's road signs.
- Synthetic occlusions are approximations for experimentation; they do not replace testing on real occlusion data.
- GPU is used automatically when available. CPU training is supported but may take longer.
