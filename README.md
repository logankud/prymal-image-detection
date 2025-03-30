
# Prymal Pouch YOLOv8 Training

This repo fine-tunes a YOLOv8 model to detect the Prymal pouch in generated and real product images.

## 📁 Directory Structure

```
yolo_pouch_detector/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
├── train.py
├── .gitignore
└── README.md
```

## 📦 Setup

```bash
pip install ultralytics mlflow
```

## 🚀 Train

```bash
python train.py
```

## 🧪 Outputs
Trained weights are saved under:
```
runs/train/prymal_pouch_detector/weights/
```

## 📊 MLflow
This project tracks:
- Model loss, precision, recall, mAP
- Training artifacts (checkpoints)

Start the MLflow UI:

```bash
mlflow ui
```

Then visit: http://localhost:5000
