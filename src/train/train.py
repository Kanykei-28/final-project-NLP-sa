from __future__ import annotations
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src.utils.paths import RAW_DIR, MODELS_DIR, METRICS_DIR
from src.utils.text_preprocess import basic_clean
from src.utils.constants import TEXT_COL, TARGET_COL, LABEL_MAP, VALID_LABELS

@dataclass
class TrainConfig:
    # Data
    train_csv: str = "train.csv"
    text_col: str = TEXT_COL
    target_col: str = TARGET_COL
    # Split
    test_size: float = 0.2
    random_state: int = 42
    # Vectorizer iwth tuned hyperparameters
    ngram_min: int = 1
    ngram_max: int = 2
    max_features: int = 200_000
    max_df: float = 0.9
    min_df: int = 1
    sublinear_tf: bool = True
    # Model with tuned hyperparameters
    C: float = 1.0
    class_weight: str | None = None
    max_iter: int = 5000
    # Outputs
    model_name: str = "tfidf_linearsvc.joblib"


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(cfg: TrainConfig) -> pd.DataFrame:
    path = RAW_DIR / cfg.train_csv
    if not path.exists():
        raise FileNotFoundError(
            f"Training file not found: {path}. "
            f"Expected it at data/raw/{cfg.train_csv} (and mounted in Docker)."
        )
    df = pd.read_csv(path)

    #validation
    for col in [cfg.text_col, cfg.target_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Found: {df.columns.tolist()}")

    if df[cfg.text_col].isna().any():
        raise ValueError("Found missing values in text column.")
    if df[cfg.target_col].isna().any():
        raise ValueError("Found missing values in target column.")
    # Nnomalizing the target
    df[cfg.target_col] = df[cfg.target_col].astype(str).str.lower().str.strip()
    valid = VALID_LABELS
    bad = set(df[cfg.target_col].unique()) - valid
    if bad:
        raise ValueError(f"Unexpected labels in target column: {bad}. Expected: {valid}")
    return df


def build_pipeline(cfg: TrainConfig) -> Pipeline:
    tfidf = TfidfVectorizer(
        preprocessor=basic_clean,  
        lowercase=False,           
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        max_features=cfg.max_features,
        max_df=cfg.max_df,
        min_df=cfg.min_df,
        sublinear_tf=cfg.sublinear_tf,)
    clf = LinearSVC(
        C=cfg.C,
        class_weight=cfg.class_weight,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        dual="auto",)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    cfg = TrainConfig(
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        random_state=int(os.getenv("SEED", "42")),
        C=float(os.getenv("SVC_C", "1.0")),
        max_iter=int(os.getenv("SVC_MAX_ITER", "5000")),
        max_features=int(os.getenv("MAX_FEATURES", "200000")),)

    ensure_dirs()
    df = load_data(cfg)
    X = df[cfg.text_col].astype(str).values
    y = df[cfg.target_col].map(LABEL_MAP).values
    X_tr, X_val, y_tr, y_val = train_test_split(X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,)

    pipe = build_pipeline(cfg)
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, pred)

    # saving the model
    import joblib  
    model_path = MODELS_DIR / cfg.model_name
    joblib.dump(pipe, model_path)

    # saving metrics
    report = classification_report(y_val, pred, target_names=[k for k,v in sorted(LABEL_MAP.items(), key=lambda x: x[1])], output_dict=True)
    cm = confusion_matrix(y_val, pred).tolist()
    run_info = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "TF-IDF + LinearSVC",
        "val_accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": report,
        "config": asdict(cfg),
        "data": {
            "n_rows": int(df.shape[0]),
            "train_size": int(len(X_tr)),
            "val_size": int(len(X_val)),
        },
    }
    metrics_path = METRICS_DIR / "train_metrics.json"
    save_json(metrics_path, run_info)
    print(f"Validation accuracy is: {acc:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()