"""
pred.py

CSC311 Project - Final prediction script
Relies on:
  - preprocess_config.json  (written by preprocessing / training script)
  - xgb_export.json         (XGBoost model dump from training script)

Functions:
  - predict(row):      predict label for a single row (pandas Series)
  - predict_all(path): predict labels for every row in a CSV file
"""

import sys
import json
import re

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Load preprocessing config
# ------------------------------------------------------------

with open("preprocess_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

CLASS_NAMES = CONFIG["class_names"]
LABEL_TO_INT = CONFIG["label_to_int"]
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

RATING_COLS_ORIG = CONFIG["rating_cols_original"]      # 4 original rating columns (strings)
RATING_LEVELS = CONFIG["rating_levels"]                # [1,2,3,4,5]
RATING_IMPUTE_VALUES = CONFIG["rating_impute_values"]  # "<col>_num" -> mode int
RATING_ONEHOT_COLS = CONFIG["rating_onehot_cols"]      # 20 one-hot col names

BEST_COL = CONFIG["best_col"]
BEST_TASK_CLASSES = CONFIG["best_task_classes"]        # 8 options

TOPIC_NAMES = CONFIG["topic_names"]                    # ['code_topic', ...]
TOPIC_KEYWORDS = CONFIG["topic_keywords"]              # topic -> list of words

TEXT_COLS = CONFIG["text_cols"]


# ------------------------------------------------------------
# 2. Helper functions: mirror preprocessing.py logic
# ------------------------------------------------------------

def extract_rating(x):
    """
    Extract the leading integer from a rating string like "3 - Often".
    Returns float or NaN.
    """
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.match(r"^(\d+)", s)
    return float(m.group(1)) if m else np.nan


def split_multiselect_cell(x):
    """
    Split a multi-select field where commas may appear inside parentheses,
    e.g. "Drafting professional text (e.g., emails, résumés)".
    We protect commas inside (...) first, then split.
    """
    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s:
        return []

    # Protect commas inside parentheses
    def protect_commas(m):
        inside = m.group(1)
        inside = inside.replace(",", "###COMMA###")
        return "(" + inside + ")"

    protected = re.sub(r"\(([^()]*)\)", protect_commas, s)

    # Split on top-level commas
    parts = [p.strip() for p in protected.split(",")]

    # Restore commas
    result = []
    for p in parts:
        if not p:
            continue
        p = p.replace("###COMMA###", ",")
        result.append(p)

    return result


def normalize_text(s: str) -> str:
    """
    Lowercase, remove non-alphanumeric characters (except spaces),
    and collapse multiple spaces.
    """
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def combine_text_row(row: pd.Series) -> str:
    """
    Combine the free-text fields into a single string.
    Missing values are treated as empty strings.
    """
    parts = []
    for col in TEXT_COLS:
        val = row.get(col, "")
        if pd.isna(val):
            val = ""
        parts.append(str(val))
    return " ".join(parts)


def text_to_topic_vector(
    text: str,
    topic_keywords_sets: dict,
    topic_names: list,
) -> np.ndarray:
    """
    For normalized text, produce a vector of length len(topic_names),
    with binary indicators: 1 if any keyword for that topic appears, else 0.
    """
    words = set(text.split())
    vec = []
    for topic in topic_names:
        kws = topic_keywords_sets[topic]
        val = 1.0 if any(w in words for w in kws) else 0.0
        vec.append(val)
    return np.array(vec, dtype=float)


# ------------------------------------------------------------
# 3. Transform a dataframe into the FINAL 33-D feature matrix
# ------------------------------------------------------------

def transform_df_to_features(df: pd.DataFrame) -> np.ndarray:
    """
    Rebuild the same 33 features as in preprocessing.py using ONLY
    the information stored in preprocess_config.json.

      X = [ratings one-hot (20) | best multi-select (8) | topics (5)]

    Returns:
      X: numpy array of shape (N, 33)
    """
    df = df.copy()

    # 3.1 Ratings -> numeric + one-hot
    for col in RATING_COLS_ORIG:
        num_col = col + "_num"

        # Extract leading integer rating
        df[num_col] = df[col].apply(extract_rating)

        # Impute missing with stored mode
        mode_val = RATING_IMPUTE_VALUES[num_col]
        df[num_col] = df[num_col].fillna(mode_val).astype(int)

        # One-hot encode
        for r in RATING_LEVELS:
            oh_col = f"{num_col}=={r}"
            df[oh_col] = (df[num_col] == r).astype(float)

    ratings_matrix = df[RATING_ONEHOT_COLS].to_numpy(dtype=float)

    # 3.2 Multi-select "best tasks" -> 8 binary features
    best_lists = df[BEST_COL].apply(split_multiselect_cell).tolist()

    best_index = {name: i for i, name in enumerate(BEST_TASK_CLASSES)}
    N = len(df)
    K_best = len(BEST_TASK_CLASSES)
    best_mat = np.zeros((N, K_best), dtype=float)

    for i, items in enumerate(best_lists):
        for item in items:
            j = best_index.get(item)
            if j is not None:
                best_mat[i, j] = 1.0

    # 3.3 Topic keyword buckets -> 5 binary features
    df["combined_text"] = df.apply(combine_text_row, axis=1)
    df["combined_text_norm"] = df["combined_text"].apply(normalize_text)

    topic_keywords_sets = {k: set(v) for k, v in TOPIC_KEYWORDS.items()}

    topic_feats = np.vstack([
        text_to_topic_vector(t, topic_keywords_sets, TOPIC_NAMES)
        for t in df["combined_text_norm"]
    ])

    # 3.4 Final X
    X = np.hstack([ratings_matrix, best_mat, topic_feats])
    return X


# ------------------------------------------------------------
# 4. Manual XGBoost inference using exported JSON trees
# ------------------------------------------------------------

def load_xgb_model_dump(path: str):
    """
    Load the JSON produced by the training script, which should look like:

      {
        "num_class": 3,
        "n_features": 33,
        "trees": [ <tree0_json>, <tree1_json>, ... ]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    return model


def traverse_tree(tree: dict, x_row: np.ndarray) -> float:
    """
    Traverse a single decision tree for one feature row x_row.

    XGBoost JSON node format:
      - Internal: "nodeid", "split", "split_condition",
                  "yes", "no", "missing", "children"
      - Leaf:     "nodeid", "leaf"
    """
    node = tree
    while True:
        if "leaf" in node:
            return float(node["leaf"])

        # feature index comes as "f12" -> 12
        split_feat = node["split"]
        fid = int(split_feat[1:])  # remove 'f'
        threshold = float(node["split_condition"])

        val = x_row[fid]

        if np.isnan(val):
            next_id = node["missing"]
        else:
            if val < threshold:
                next_id = node["yes"]
            else:
                next_id = node["no"]

        # find the child whose nodeid == next_id
        children = node.get("children", [])
        child = None
        for ch in children:
            if ch["nodeid"] == next_id:
                child = ch
                break

        if child is None:
            # Should not happen; safe fallback
            return 0.0

        node = child


def predict_xgb_from_dump(X: np.ndarray, model_dump: dict) -> np.ndarray:
    """
    Manual inference for a multi-class XGBoost model using the JSON dump.

    Assumes trees are ordered such that tree i contributes to class
    (i % num_class).

    For each sample:
      - Sum leaf values per class
      - Apply softmax over classes
      - Return argmax class index
    """
    num_class = int(model_dump["num_class"])
    trees = model_dump["trees"]

    N, _ = X.shape
    scores = np.zeros((N, num_class), dtype=float)

    for tree_idx, tree in enumerate(trees):
        class_id = tree_idx % num_class
        for i in range(N):
            leaf_val = traverse_tree(tree, X[i])
            scores[i, class_id] += leaf_val

    # Softmax: exp(scores) / sum(exp(scores))
    max_scores = scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    preds = np.argmax(probs, axis=1)
    return preds


# Load model dump once at import time so predict() is fast-ish
_XGB_DUMP = load_xgb_model_dump("xgb_export.json")


# ------------------------------------------------------------
# 5. API required by the assignment: predict + predict_all
# ------------------------------------------------------------

def predict(row: pd.Series) -> str:
    """
    Helper function to make prediction for a given input row (pandas Series).

    This replaces the original random baseline. It:
      - Builds the 33-dimensional feature vector for this row.
      - Runs manual XGBoost inference using xgb_export.json.
      - Returns one of: 'ChatGPT', 'Claude', 'Gemini'.
    """
    # Create a single-row DataFrame so we can reuse transform_df_to_features
    df_one = pd.DataFrame([row])
    X_one = transform_df_to_features(df_one)  # shape (1, 33)

    pred_int = predict_xgb_from_dump(X_one, _XGB_DUMP)[0]
    label_name = INT_TO_LABEL[int(pred_int)]
    return label_name


def predict_all(filename: str):
    """
    Make predictions for the data in filename.

    Returns:
        A list of predicted label strings, one per row.
    """
    df = pd.read_csv(filename)

    predictions = []
    for _, row in df.iterrows():
        pred_label = predict(row)
        predictions.append(pred_label)

    return predictions


# ------------------------------------------------------------
# 6. Main: CLI entrypoint - print labels only
# ------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python pred.py <test_csv_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    preds = predict_all(csv_path)

    # Print one label per line ONLY
    for label in preds:
        print(label)

if __name__ == "__main__":
    main()
