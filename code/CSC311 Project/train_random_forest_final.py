"""
train_random_forest.py

训练阶段脚本：
  1. 读取 training_data_clean.csv
  2. 调用 preprocessing.build_features 构造特征 X, 标签 y
  3. 按 student_id 做 3-way 划分：train / val / test
  4. 在 train 上训练 Random Forest，用 val 选超参数
  5. 使用最优模型在 test 上做一次最终评估
  6. 导出：
       - preprocess_config.json（给 pred.py 复现预处理）
       - rf_export.json（给 pred.py 复现随机森林）
"""

import json
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import preprocessing_new  # 你组员写的 preprocessing.py

# ----------------------------------------------------
# 1. 一些路径（如果文件名不同，这里改一下）
# ----------------------------------------------------
TRAIN_CSV_PATH = "training_data_clean.csv"
PREPROCESS_CONFIG_PATH = "preprocess_config.json"
TOPIC_KEYWORDS_PATH = "topic_keywords_final.json"
RF_EXPORT_PATH = "rf_export.json"


# ----------------------------------------------------
# 2. 导出随机森林结构到 JSON（给 pred.py 使用）
# ----------------------------------------------------
def export_rf_to_json(rf: RandomForestClassifier,
                      filename: str = RF_EXPORT_PATH):
    """
    把 sklearn 的 RandomForestClassifier 导出成纯参数形式，
    方便在 pred.py 里用 numpy 手写推理。
    """
    forest = []

    for estimator in rf.estimators_:
        tree = estimator.tree_
        tree_dict = {
            # 左右孩子下标，-1 代表叶子
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            # 每个节点用哪个特征做划分，-2 表示叶子
            "feature": tree.feature.tolist(),
            # 划分阈值
            "threshold": tree.threshold.tolist(),
            # 每个节点的类别计数（shape: [n_nodes, n_classes]）
            "values": tree.value.squeeze(axis=1).tolist(),
        }
        forest.append(tree_dict)

    model = {
        "n_classes": rf.n_classes_,
        "n_trees": len(rf.estimators_),
        "trees": forest,
    }

    with open(filename, "w") as f:
        json.dump(model, f)

    print(f"[INFO] Exported Random Forest to {filename}")


# ----------------------------------------------------
# 3. 训练主流程
# ----------------------------------------------------
def main():
    # 3.1 读入清洗后的训练数据
    print(f"[INFO] Loading training data from {TRAIN_CSV_PATH} ...")
    df = pd.read_csv(TRAIN_CSV_PATH)

    # 3.2 读取 topic_keywords
    # 你们之前在 preprocessing.py 里就是从 topic_keywords_final.json 读的
    print(f"[INFO] Loading topic keywords from {TOPIC_KEYWORDS_PATH} ...")
    with open(TOPIC_KEYWORDS_PATH, "r") as f:
        topic_keywords = json.load(f)

    # 3.3 用 preprocessing.build_features 构造 X, y, config
    print("[INFO] Building features with preprocessing.build_features ...")
    X, y, config = preprocessing_new.build_features(df, topic_keywords)

    print(f"[INFO] Feature matrix shape (all): {X.shape}")  # (N, D)
    print(f"[INFO] Label vector shape (all): {y.shape}")
    print(f"[INFO] Class names: {config['class_names']}")

    # 3.4 保存 preprocess_config.json（给 pred.py 使用）
    with open(PREPROCESS_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved preprocess config to {PREPROCESS_CONFIG_PATH}")

    # 3.5 按 student_id 做 3-way 分割：train / val / test
    student_ids = df[preprocessing_new.STUDENT_ID_COL].to_numpy()

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = preprocessing_new.grouped_train_val_test_split(
        X,
        y,
        student_ids,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Val shape:   {X_val.shape}")
    print(f"[INFO] Test shape:  {X_test.shape}")

    # ------------------------------------------------
    # 4. 手动网格搜索 Random Forest 超参数（用 val 选最优）
    # ------------------------------------------------
    param_grid = {
        "n_estimators": [100, 150, 200, 250],  # 树的数量
        "max_depth": [10, 15, 20, 25, 30, None],  # 每棵树最大深度
        "min_samples_leaf": [1, 2, 4, 6, 8],  # 叶子最小样本数（控制过拟合）
        "max_features": ["sqrt", "log2", None, 0.5],  # 每次划分考虑的特征数
    }

    all_param_combos = list(itertools.product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_leaf"],
        param_grid["max_features"],
    ))
    print(f"[INFO] Total hyperparameter combinations: {len(all_param_combos)}")

    best_val_acc = -1.0
    best_params = None
    best_model = None

    for (n_estimators, max_depth, min_samples_leaf,
         max_features) in all_param_combos:
        print(
            f"\n[INFO] Training RF with "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_leaf={min_samples_leaf}, max_features={max_features}"
        )

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )

        # 在 training set 上训练
        rf.fit(X_train, y_train)

        # 看看 training accuracy（用来监控是否严重 overfit）
        y_train_pred = rf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"[INFO]   Train accuracy: {train_acc:.4f}")

        # 在 validation set 上评估，用来挑选最优超参数
        y_val_pred = rf.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"[INFO]   Val accuracy:   {val_acc:.4f}")

        # 记录最好的一组参数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
            best_model = rf
            print("[INFO]   --> New best model found!")

    print("\n==============================")
    print("[RESULT] Best validation accuracy:", best_val_acc)
    print("[RESULT] Best hyperparameters:", best_params)
    print("==============================\n")

    # ------------------------------------------------
    # 5. 使用最佳模型，打印更详细的指标
    # ------------------------------------------------
    # Training set
    # y_train_pred = best_model.predict(X_train)
    # train_acc = accuracy_score(y_train, y_train_pred)
    # print("[FINAL] Training accuracy:", train_acc)
    # print("\n[FINAL] Classification report (TRAIN):")
    # print(classification_report(y_train, y_train_pred, target_names=config["class_names"]))
    # Training set
    y_train_pred = best_model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print("[FINAL] Training accuracy:", f"{train_acc:.4f}")

    train_report = classification_report(
        y_train, y_train_pred,
        target_names=config["class_names"],
        output_dict=True
    )

    print("\n[FINAL] Classification report (TRAIN):")
    print("Class        Precision  Recall   F1-score  Support")
    for cls in config["class_names"]:
        p = train_report[cls]["precision"]
        r = train_report[cls]["recall"]
        f = train_report[cls]["f1-score"]
        s = train_report[cls]["support"]
        print(f"{cls:12s} {p:.4f}     {r:.4f}   {f:.4f}    {s}")

    macro_p = train_report["macro avg"]["precision"]
    macro_r = train_report["macro avg"]["recall"]
    macro_f = train_report["macro avg"]["f1-score"]
    print(f"\nMacro avg    {macro_p:.4f}     {macro_r:.4f}   {macro_f:.4f}")

    # Validation set
    # y_val_pred = best_model.predict(X_val)
    # val_acc = accuracy_score(y_val, y_val_pred)
    # print("\n[FINAL] Validation accuracy:", val_acc)
    # print("\n[FINAL] Classification report (VAL):")
    # print(classification_report(y_val, y_val_pred,
    #                             target_names=config["class_names"]))
    # print("\n[FINAL] Confusion matrix (VAL):")
    # print(confusion_matrix(y_val, y_val_pred))

    # Validation set
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print("\n[FINAL] Validation accuracy:", f"{val_acc:.4f}")

    val_report = classification_report(
        y_val, y_val_pred,
        target_names=config["class_names"],
        output_dict=True
    )

    print("\n[FINAL] Classification report (VAL):")
    print("Class        Precision  Recall   F1-score  Support")
    for cls in config["class_names"]:
        p = val_report[cls]["precision"]
        r = val_report[cls]["recall"]
        f = val_report[cls]["f1-score"]
        s = val_report[cls]["support"]
        print(f"{cls:12s} {p:.4f}     {r:.4f}   {f:.4f}    {s}")

    macro_p = val_report["macro avg"]["precision"]
    macro_r = val_report["macro avg"]["recall"]
    macro_f = val_report["macro avg"]["f1-score"]
    print(f"\nMacro avg    {macro_p:.4f}     {macro_r:.4f}   {macro_f:.4f}")


# # Test set（只在最终评估时使用一次）
#     y_test_pred = best_model.predict(X_test)
#     test_acc = accuracy_score(y_test, y_test_pred)
#     print("\n[FINAL] Test accuracy:", test_acc)
#     print("\n[FINAL] Classification report (TEST):")
#     print(classification_report(y_test, y_test_pred,
#                                 target_names=config["class_names"]))
#     print("\n[FINAL] Confusion matrix (TEST):")
#     print(confusion_matrix(y_test, y_test_pred))
#
#     # ------------------------------------------------
#     # 6. 导出最佳随机森林模型到 JSON，给 pred.py 用
#     # ------------------------------------------------
#     export_rf_to_json(best_model, filename=RF_EXPORT_PATH)
#     print("[INFO] Training and export finished.")

# ---- Test set ----
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("\n[FINAL] Test accuracy:", f"{test_acc:.4f}")

    test_report = classification_report(
        y_test, y_test_pred,
        target_names=config["class_names"],
        output_dict=True
    )

    print("\n[FINAL] Classification report (TEST):")
    print("Class        Precision  Recall   F1-score  Support")
    for cls in config["class_names"]:
        p = test_report[cls]["precision"]
        r = test_report[cls]["recall"]
        f = test_report[cls]["f1-score"]
        s = test_report[cls]["support"]
        print(f"{cls:12s} {p:.4f}     {r:.4f}   {f:.4f}    {s}")

    macro_p = test_report["macro avg"]["precision"]
    macro_r = test_report["macro avg"]["recall"]
    macro_f = test_report["macro avg"]["f1-score"]
    print(f"\nMacro avg    {macro_p:.4f}     {macro_r:.4f}   {macro_f:.4f}")

    print("\n[FINAL] Confusion matrix (TEST):")
    print(confusion_matrix(y_test, y_test_pred))



if __name__ == "__main__":
    main()
