import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,                 # NEW
    precision_recall_fscore_support,  # NEW
)
from xgboost import XGBClassifier

from preprocessing import (
    build_features,
    grouped_train_val_test_split,
    STUDENT_ID_COL,
)

"""
Parameter searched in 7.5k random search run:
    "max_depth":         int(rng.choice([2, 3, 4, 5, 6])),
    "learning_rate":     float(rng.choice([0.02, 0.03, 0.05, 0.07, 0.1, 0.15])),
    "n_estimators":      int(rng.choice([80, 120, 160, 220, 280, 340, 400])),
    "subsample":         float(rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])),
    "colsample_bytree":  float(rng.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
    "reg_lambda":        float(rng.choice([0.1, 0.3, 1.0, 3.0, 5.0, 8.0, 10.0])),
    "min_child_weight":  int(rng.choice([1, 2, 3, 4, 5])),

Best parameters found in 7.5k random search run:
    {'max_depth': 5, 'learning_rate': 0.07, 'n_estimators': 120,
     'subsample': 0.7, 'colsample_bytree': 0.5, 'reg_lambda': 10.0,
     'min_child_weight': 5}

Parameters searched in 200 random search run:
    "max_depth":        int(rng.choice([4, 5, 6])),
    "learning_rate":    float(rng.choice([0.05, 0.06, 0.07, 0.08, 0.09])),
    "n_estimators":     int(rng.choice([90, 120, 150, 180])),
    "subsample":        float(rng.choice([0.6, 0.7, 0.8])),
    "colsample_bytree": float(rng.choice([0.4, 0.5, 0.6])),
    "reg_lambda":       float(rng.choice([5.0, 10.0, 15.0])),
    "min_child_weight": int(rng.choice([3, 5, 7])),

Best parameters found in 200 random search run:
    {'max_depth': 5, 'learning_rate': 0.08, 'n_estimators': 150,
     'subsample': 0.7, 'colsample_bytree': 0.5, 'reg_lambda': 15.0,
     'min_child_weight': 5}

For the FINAL model, we use the 200-search best parameters.
"""

# ---- Final chosen hyperparameters (from the 200-run local search) ----
BEST_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.08,
    "n_estimators": 150,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_lambda": 15.0,
    "min_child_weight": 5,
}


def main():
    # ----------------------------------------------------------
    # 1. Load data + build final features (33-dim X)
    # ----------------------------------------------------------
    df = pd.read_csv("training_data_clean.csv")

    with open("topic_keywords_final.json", "r", encoding="utf-8") as f:
        topic_keywords = json.load(f)

    X, y, config = build_features(df, topic_keywords)
    student_ids = df[STUDENT_ID_COL].to_numpy()

    # Grouped 3-way split
    X_train, y_train, X_val, y_val, X_test, y_test = grouped_train_val_test_split(
        X,
        y,
        student_ids,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    num_classes = len(np.unique(y))

    # Save config for pred.py
    with open("preprocess_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("Saved preprocess_config.json")
    print("X_all shape:", X.shape, " y shape:", y.shape)
    print("Train:", X_train.shape, " Val:", X_val.shape, " Test:", X_test.shape)

    # ----------------------------------------------------------
    # 2. Train ONE XGBoost model with the 200-search best params
    # ----------------------------------------------------------
    base_params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,   # use 1 + tree_method="exact" if you want strict determinism
    }
    base_params.update(BEST_PARAMS)

    model = XGBClassifier(**base_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ----------------------------------------------------------
    # 3. Compute predictions + metrics
    # ----------------------------------------------------------
    # Train
    train_proba = model.predict_proba(X_train)
    y_pred_train = np.argmax(train_proba, axis=1)
    train_acc = accuracy_score(y_train, y_pred_train)

    # Val
    val_proba = model.predict_proba(X_val)
    y_pred_val = np.argmax(val_proba, axis=1)
    val_acc = accuracy_score(y_val, y_pred_val)

    # Test
    test_proba = model.predict_proba(X_test)
    y_pred_test = np.argmax(test_proba, axis=1)
    test_acc = accuracy_score(y_test, y_pred_test)

    # --- NEW: macro precision / recall / F1 for val + test ---
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
        y_val, y_pred_val, average="macro"
    )
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average="macro"
    )

    # --- NEW: confusion matrix for test ---
    cm_test = confusion_matrix(y_test, y_pred_test)

    # ----------------------------------------------------------
    # 3a. Print results in a way that matches your tables
    # ----------------------------------------------------------
    print("\n==================== RESULTS (200-search best XGBoost) ====================")
    print("Best params (fixed):", BEST_PARAMS)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Val accuracy:   {val_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    # ---- Table 1-style Validation metrics ----
    print("\n=== Table 1 – Validation Performance (XGBoost) ===")
    print(f"Accuracy       : {val_acc:.4f}")
    print(f"Macro Precision: {val_prec:.4f}")
    print(f"Macro Recall   : {val_rec:.4f}")
    print(f"Macro F1       : {val_f1:.4f}")

    # ---- Table 3-style Test metrics (single final model) ----
    print("\n=== Table 3 – Test Performance (XGBoost, final single-model estimate) ===")
    print(f"Accuracy       : {test_acc:.4f}")
    print(f"Macro Precision: {test_prec:.4f}")
    print(f"Macro Recall   : {test_rec:.4f}")
    print(f"Macro F1       : {test_f1:.4f}")

    # ---- Confusion matrix for test set (for Table 2) ----
    class_names = config["class_names"]
    print("\n=== Table 2 – Confusion Matrix on Test Set (rows = true, cols = pred) ===")
    print("Classes (index order):", class_names)
    print(cm_test)

    # Optional: keep the detailed classification reports too
    print("\nClassification report (Validation):")
    print(classification_report(
        y_val, y_pred_val, digits=3, target_names=class_names
    ))

    print("\nClassification report (Test):")
    print(classification_report(
        y_test, y_pred_test, digits=3, target_names=class_names
    ))

    # ----------------------------------------------------------
    # 4. Retrain on FULL data and export JSON dump for pred.py
    # ----------------------------------------------------------
    print("\nRetraining on FULL dataset with the same 200-search best params...")
    model_full = XGBClassifier(**base_params)
    model_full.fit(X, y)

    booster = model_full.get_booster()
    tree_json_strings = booster.get_dump(dump_format="json")

    export = {
        "num_class": int(num_classes),
        "n_features": int(X.shape[1]),
        "trees": [json.loads(s) for s in tree_json_strings],
    }

    with open("xgb_export.json", "w", encoding="utf-8") as f:
        json.dump(export, f)

    print(f"Saved xgb_export.json with {len(tree_json_strings)} trees.")


if __name__ == "__main__":
    main()
