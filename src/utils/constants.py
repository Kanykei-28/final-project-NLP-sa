from __future__ import annotations

# Column names 
TEXT_COL = "review"
TARGET_COL = "sentiment"
# Labels
LABEL_MAP = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}
VALID_LABELS = set(LABEL_MAP.keys())

