
import argparse
from ultralytics import YOLO
import mlflow

def train(args):
    mlflow.set_experiment("Prymal YOLOv8 Detection")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "data_yaml": args.data
        })

        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.run_name,
            project=args.project
        )

        if hasattr(results, 'metrics') and results.metrics:
            mlflow.log_metrics({
                "precision": results.metrics.get("metrics/precision(B)", 0),
                "recall": results.metrics.get("metrics/recall(B)", 0),
                "mAP50": results.metrics.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.metrics.get("metrics/mAP50-95(B)", 0),
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.yaml", help="YOLOv8 model config")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--run_name", default="prymal_pouch_detector", help="Run name for output and MLflow")
    parser.add_argument("--project", default="runs/train", help="Output directory for YOLOv8")

    args = parser.parse_args()
    train(args)
