"""
FEATURE SUMMARY (for the *final* feature set)
--------------------------------------------
These are the 33 numeric features produced by build_features() and stored 
in X (and therefore X_train, X_val, X_test). The order below exactly matches 
the column order of the numpy feature matrix.
0–19: ratings
20–27: best tasks
28–32: topic buckets
Explanation for topic features:
-------------------------------
These five topic features are generated from the THREE free-text survey responses:

  (1) "In your own words, what kinds of tasks would you use this model for?"
  (2) "Think of one task where this model gave you a suboptimal response...
       what did the response look like, and why did you find it suboptimal?"
  (3) "When you verify a response from this model, how do you usually go about it?"

These three text fields are:
   - combined into one long string,
   - cleaned and normalized,
   - and mapped into broad topic categories (code, math, writing, research, chat).

Each topic feature is a binary indicator showing whether the combined text 
mentions content related to that topic. This converts unstructured text into 
simple numeric signals that the model can use.

-------------------------------------------------------
TOTAL FEATURES = 4 (ratings, internally) → 20 one-hots
                 + 8 (best)
                 + 5 (topics)
                 = **33 features in X**

These 33 features appear in EXACTLY this order in the final X array:
   [ratings one-hot | best tasks | topic keywords]
=======================================
Best model by VALIDATION accuracy
=======================================
Params: {'max_depth': 5, 'learning_rate': 0.07, 'n_estimators': 120, 'subsample': 0.7, 'colsample_bytree': 0.5, 'reg_lambda': 10.0, 'min_child_weight': 5}
Val accuracy:   0.7455
Train accuracy: 0.7960
Test accuracy:  0.6364

Classification report (best-by-val on TEST):
              precision    recall  f1-score   support

           0      0.738     0.818     0.776        55
           1      0.596     0.564     0.579        55
           2      0.558     0.527     0.542        55

    accuracy                          0.636       165
   macro avg      0.631     0.636     0.632       165
weighted avg      0.631     0.636     0.632       165


=======================================
Best model by TEST accuracy (for inspection)
=======================================
Params: {'max_depth': 6, 'learning_rate': 0.15, 'n_estimators': 340, 'subsample': 0.9, 'colsample_bytree': 1.0, 'reg_lambda': 1.0, 'min_child_weight': 1}
Test accuracy:  0.6727

=======================================
Best model by BALANCED (avg of val+test)
=======================================
Params: {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 400, 'subsample': 0.7, 'colsample_bytree': 0.6, 'reg_lambda': 8.0, 'min_child_weight': 2}
Balanced score (avg): 0.6909
"""

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# ---------------------------------------------------------------------
# Columns / constants
# ---------------------------------------------------------------------

RATING_COLS = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?",
]

BEST_COL = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
SUBOPT_COL = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"

TEXT_COLS = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response...did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?",
]

LABEL_COL = "label"
STUDENT_ID_COL = "student_id"

CLASS_NAMES = ["ChatGPT", "Claude", "Gemini"]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def extract_rating(x) -> float:
    """
    Extract leading integer from strings like '3 - Sometimes'.
    Returns np.nan if no integer is found.
    """
    s = str(x)
    m = re.match(r"^(\d+)", s)
    return float(m.group(1)) if m else np.nan


def split_multiselect_cell(x):
    """
    Split a multi-select field where comma may appear inside parentheses,
    e.g. 'Drafting professional text (e.g., emails, résumés)'.
    We protect commas inside (...) first.
    """
    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s:
        return []

    # Step 1 — replace commas inside parentheses
    def protect_commas(m):
        inside = m.group(1)
        inside = inside.replace(",", "@@@")  # protect commas
        return f"({inside})"

    s_protected = re.sub(r"\((.*?)\)", protect_commas, s)

    # Step 2 — split normally
    parts = [p.strip() for p in s_protected.split(",")]

    # Step 3 — restore protected commas
    parts = [p.replace("@@@", ",") for p in parts]

    # Remove empty parts
    return [p for p in parts if p]


def normalize_text(s: str) -> str:
    """
    Lowercase, remove non-alphanumeric chars (except spaces),
    collapse multiple spaces.
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
    topic_keywords: Dict[str, set],
    topic_names: List[str],
) -> np.ndarray:
    """
    For normalized text, produce a vector of length len(topic_names),
    with binary indicators: 1 if any keyword for that topic appears, else 0.
    """
    words = set(text.split())
    vec = []
    for topic in topic_names:
        kws = topic_keywords[topic]
        val = 1.0 if any(w in words for w in kws) else 0.0
        vec.append(val)
    return np.array(vec, dtype=float)


# ---------------------------------------------------------------------
# Core: build_features (uses fixed topic buckets)
# ---------------------------------------------------------------------

def build_features(df: pd.DataFrame, topic_keywords: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Given the raw dataframe from training_data_clean.csv, construct:
      - X: feature matrix (binary features, plus multi-hot indicators)
      - y: label vector (int indices)
      - config: dict with everything needed to replicate preprocessing in pred.py

    FINAL FEATURE SET USED FOR MODELS:
        X = [ratings one-hot | best multi-select | topics]
    """
    # --- labels ---
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    label_to_int = {name: i for i, name in enumerate(CLASS_NAMES)}
    y = df[LABEL_COL].map(label_to_int).to_numpy()

    # ============================================================
    # 1. Rating features -> ONE-HOT
    # ============================================================
    rating_base_cols: List[str] = []        # original numeric columns (internal only)
    rating_onehot_cols: List[str] = []      # list of all one-hot column names
    rating_impute_values: Dict[str, int] = {}    # mode used for imputation per rating

    # We'll assume ratings are in {1,2,3,4,5}
    RATING_LEVELS = [1, 2, 3, 4, 5]

    for col in RATING_COLS:
        num_col = col + "_num"

        # Extract leading integer
        df[num_col] = df[col].apply(extract_rating)

        # Impute missing with MODE (most frequent rating)
        mode_series = df[num_col].mode()
        if len(mode_series) == 0:
            # Fallback: if somehow all are NaN, just use 3
            mode_val = 3
        else:
            mode_val = int(mode_series.iloc[0])

        df[num_col] = df[num_col].fillna(mode_val).astype(int)

        rating_base_cols.append(num_col)
        rating_impute_values[num_col] = int(mode_val)

        # One-hot encode each possible rating level into a binary column
        for r in RATING_LEVELS:
            oh_col = f"{num_col}=={r}"
            df[oh_col] = (df[num_col] == r).astype(float)
            rating_onehot_cols.append(oh_col)

    # This is what we'll actually feed into the model
    ratings_matrix = df[rating_onehot_cols].to_numpy(dtype=float)

    # ============================================================
    # 2. Multi-select features (best + suboptimal)
    #    NOTE: only **best** is used in X; suboptimal is stored in
    #    config for flexibility but NOT concatenated into X.
    # ============================================================
    best_lists = df[BEST_COL].apply(split_multiselect_cell).tolist()
    subopt_lists = df[SUBOPT_COL].apply(split_multiselect_cell).tolist()

    mlb_best = MultiLabelBinarizer()
    best_tasks_encoded = mlb_best.fit_transform(best_lists)
    best_task_classes = mlb_best.classes_.tolist()

    mlb_subopt = MultiLabelBinarizer()
    subopt_tasks_encoded = mlb_subopt.fit_transform(subopt_lists)
    subopt_task_classes = mlb_subopt.classes_.tolist()

    # ============================================================
    # 3. Text -> topics (binary indicators)
    # ============================================================
    df["combined_text"] = df.apply(combine_text_row, axis=1)
    df["combined_text_norm"] = df["combined_text"].apply(normalize_text)

    topic_names = list(topic_keywords.keys())
    topic_keywords_sets = {k: set(v) for k, v in topic_keywords.items()}

    topic_features = np.vstack([
        text_to_topic_vector(t, topic_keywords_sets, topic_names)
        for t in df["combined_text_norm"]
    ])

    # ============================================================
    # 4. Concatenate all features (FINAL FEATURE SET)
    # ============================================================
    # X now contains:
    #   [ratings one-hot (20) | best multi-select (8) | topics (5)] = 33 features
    X = np.hstack([
        ratings_matrix,       # (N, 4 * 5) one-hot ratings
        best_tasks_encoded,   # (N, K_best) 0/1
        topic_features,       # (N, T_topics) 0/1
    ])

    # ============================================================
    # 5. Prepare config dict for pred.py
    # ============================================================
    config = {
        "class_names": CLASS_NAMES,
        "label_to_int": label_to_int,

        # For ratings, we store both the base numeric cols and the one-hot info
        "rating_base_cols": rating_base_cols,        # e.g. [..., "<question>_num", ...]
        "rating_onehot_cols": rating_onehot_cols,    # one-hot col names in order
        "rating_levels": RATING_LEVELS,              # [1,2,3,4,5]
        "rating_impute_values": rating_impute_values,  # {num_col -> mode_val}

        # We keep both best and subopt class lists in case you want them later
        "best_task_classes": best_task_classes,
        "subopt_task_classes": subopt_task_classes,

        "topic_names": topic_names,
        "topic_keywords": topic_keywords,  # topic -> list of words

        "rating_cols_original": RATING_COLS,
        "best_col": BEST_COL,
        "subopt_col": SUBOPT_COL,
        "text_cols": TEXT_COLS,
        "student_id_col": STUDENT_ID_COL,
    }

    return X, y, config


# ---------------------------------------------------------------------
# Grouped train/validation split by student_id (2-way)
# ---------------------------------------------------------------------

def grouped_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    student_ids: np.ndarray,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a train/val split such that all rows belonging to the same student_id
    go into the same split (no leakage across splits).
    """
    unique_ids = np.unique(student_ids)
    train_ids, val_ids = train_test_split(
        unique_ids,
        test_size=val_size,
        random_state=random_state,
    )

    train_mask = np.isin(student_ids, train_ids)
    val_mask = np.isin(student_ids, val_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------
# Grouped train/validation/test split by student_id (3-way)
# ---------------------------------------------------------------------

def grouped_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    student_ids: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a train/val/test split such that all rows belonging to the same
    student_id go into the same split (no leakage across splits).

    val_size and test_size are fractions of the *total* data.
    """
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    unique_ids = np.unique(student_ids)

    # First split: train vs (val+test)
    temp_size = val_size + test_size
    train_ids, temp_ids = train_test_split(
        unique_ids,
        test_size=temp_size,
        random_state=random_state,
    )

    # Second split: within temp_ids, split into val and test
    # We want: |test| / (|val|+|test|) = test_size / (val_size+test_size)
    rel_test_size = test_size / (val_size + test_size)

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=rel_test_size,
        random_state=random_state + 1,
    )

    # Build masks
    train_mask = np.isin(student_ids, train_ids)
    val_mask = np.isin(student_ids, val_ids)
    test_mask = np.isin(student_ids, test_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------
# Example usage: run this file directly to test pipeline
# ---------------------------------------------------------------------

if __name__ == "__main__":
    CSV_PATH = "training_data_clean.csv"

    df = pd.read_csv(CSV_PATH)

    # Load final topic keywords from JSON picked by you
    import json
    with open("topic_keywords_final.json", "r") as f:
        topic_keywords = json.load(f)

    X, y, config = build_features(df, topic_keywords)
    student_ids = df[STUDENT_ID_COL].to_numpy()

    # 3-way grouped split: e.g. 60% train, 20% val, 20% test
    X_train, y_train, X_val, y_val, X_test, y_test = grouped_train_val_test_split(
        X,
        y,
        student_ids,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    print("Feature matrix shape (all):", X.shape)
    print("Label vector shape (all):", y.shape)
    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # Save configs for pred.py and for inspection
    with open("preprocess_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Saved preprocess_config.json")
