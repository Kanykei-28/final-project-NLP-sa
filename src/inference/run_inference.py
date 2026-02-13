from __future__ import annotations
import argparse
import json
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.utils.paths import MODELS_DIR, METRICS_DIR, PREDICTIONS_DIR

LABEL_MAP = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}


@dataclass
class InferenceMetrics:
    timestamp_utc: str
    model_path: str
    input_path: str
    n_rows: int
    has_ground_truth: bool
    accuracy: Optional[float]
    confusion_matrix: Optional[list]
    classification_report: Optional[dict]

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def utc_now_tag() -> str:
    # using filesystem-safe timestamp
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    df = pd.read_csv(input_csv)
    if "review" not in df.columns:
        raise ValueError(f"Missing required column 'review'. Columns: {df.columns.tolist()}")
    return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.Series]]:
    X = df["review"].astype(str)
    y = None
    if "sentiment" in df.columns:
        y = df["sentiment"].astype(str).str.lower().str.strip().map(LABEL_MAP)
        if y.isna().any():
            bad = df.loc[y.isna(), "sentiment"].unique().tolist()
            raise ValueError(
                f"Unknown target labels found: {bad}. Expected only {list(LABEL_MAP.keys())}"
            )
    return X, y

def save_predictions(df_in: pd.DataFrame, y_pred_ids, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = df_in.copy()
    out_df["predicted_label"] = [ID_TO_LABEL[int(i)] for i in y_pred_ids]
    out_df.to_csv(out_path, index=False)

def save_metrics_json(metrics: InferenceMetrics, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

def append_run_log_csv(metrics: InferenceMetrics, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": metrics.timestamp_utc,
        "model_path": metrics.model_path,
        "input_path": metrics.input_path,
        "n_rows": metrics.n_rows,
        "has_ground_truth": metrics.has_ground_truth,
        "accuracy": metrics.accuracy,
    }

    file_exists = log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    default_model_path = MODELS_DIR / "tfidf_linearsvc.joblib"
    parser = argparse.ArgumentParser(description="Run batch inference using a pretrained model.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--model", type=str, default=str(default_model_path))
    parser.add_argument("--pred-out", type=str, default="")
    parser.add_argument("--metrics-out", type=str, default="")
    args = parser.parse_args()
    input_path = Path(args.input)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train and save the model first.")

    df = load_data(input_path)
    X, y_true = split_xy(df)
    model = joblib.load(model_path)
    y_pred = model.predict(X)
    tag = utc_now_tag()
    default_pred_path = PREDICTIONS_DIR / f"predictions_{tag}.csv"
    default_metrics_path = METRICS_DIR / f"inference_metrics_{tag}.json"
    pred_out_path = Path(args.pred_out) if args.pred_out else default_pred_path
    metrics_out_path = Path(args.metrics_out) if args.metrics_out else default_metrics_path
    save_predictions(df, y_pred, pred_out_path)

    if y_true is not None:
        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(y_true, y_pred, target_names=["negative", "positive"], output_dict=True)
        print(f"Accuracy on input file: {acc:.4f}")
    else:
        acc = None
        cm = None
        report = None
        print("No ground truth column 'sentiment' found.")

    metrics = InferenceMetrics(
        timestamp_utc=utc_now_iso(),
        model_path=str(model_path),
        input_path=str(input_path),
        n_rows=int(len(df)),
        has_ground_truth=(y_true is not None),
        accuracy=acc,
        confusion_matrix=cm,
        classification_report=report,)

    save_metrics_json(metrics, metrics_out_path)
    append_run_log_csv(metrics, METRICS_DIR / "inference_runs.csv")
    print(f"Saved predictions: {pred_out_path}")
    print(f"Saved metrics: {metrics_out_path}")
    print(f"Appended run log: {METRICS_DIR / 'inference_runs.csv'}")


if __name__ == "__main__":
    main()