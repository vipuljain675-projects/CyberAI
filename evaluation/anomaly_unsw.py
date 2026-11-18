#!/usr/bin/env python3
"""Benign-only Isolation Forest and supervised/anomaly union on UNSW-NB15."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from imbalance_unsw import DATA, SEED, load_frame, make_encoder
from binary_unsw import choose_threshold

OUT = Path(__file__).resolve().parent / "reports" / "unsw_nb15_anomaly_experiment.json"


def evaluate(y, flagged, score=None):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(flagged, dtype=int)
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "attack_precision": float(p[1]), "attack_recall": float(r[1]),
        "attack_f1": float(f[1]),
        "benign_false_positive_rate": float(np.mean(pred[y == 0])) if np.any(y == 0) else None,
        "attack_roc_auc": float(roc_auc_score(y, score)) if score is not None else None,
        "attack_average_precision": float(average_precision_score(y, score)) if score is not None else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    a = load_frame(DATA / "UNSW_NB15_training-set.csv")
    b = load_frame(DATA / "UNSW_NB15_testing-set.csv")
    if len(a) == 175_341:
        train_df, test_df = a, b
    elif len(b) == 175_341:
        train_df, test_df = b, a
    else:
        raise ValueError("Unexpected UNSW partition sizes")
    y_all = (train_df.attack_cat != "Normal").astype(int).to_numpy()
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SEED, stratify=y_all
    )
    x_train, x_val, cat_idx, _ = make_encoder(train_df.iloc[train_idx], train_df.iloc[val_idx])
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    x_full, x_test, cat_idx, _ = make_encoder(train_df, test_df)
    y_test = (test_df.attack_cat != "Normal").astype(int).to_numpy()

    # Supervised comparator: class-weighted binary LightGBM; threshold set on val.
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, min_child_samples=35, colsample_bytree=0.9,
        subsample=0.9, subsample_freq=1, reg_lambda=1.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    clf.fit(x_train, y_train, categorical_feature=cat_idx,
            sample_weight=compute_sample_weight("balanced", y_train))
    p_val = clf.predict_proba(x_val)[:, list(clf.classes_).index(1)]
    clf_threshold = choose_threshold(y_val, p_val, "min_fpr_recall90")
    clf_flag_val = p_val >= clf_threshold

    # Isolation Forest sees benign training rows only. Quantile thresholds are
    # calibrated with validation benign rows, and test labels never set thresholds.
    benign_train = x_train.loc[y_train == 0]
    forest = IsolationForest(
        n_estimators=200, max_samples=512, contamination="auto",
        n_jobs=2, random_state=SEED,
    )
    forest.fit(benign_train)
    anomaly_val = -forest.score_samples(x_val)
    anomaly_test = -forest.score_samples(x_test)
    benign_val_scores = anomaly_val[y_val == 0]
    anomaly_thresholds = {
        "validation_fpr_2pct": float(np.quantile(benign_val_scores, 0.98)),
        "validation_fpr_0_5pct": float(np.quantile(benign_val_scores, 0.995)),
    }

    clf_full = lgb.LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, min_child_samples=35, colsample_bytree=0.9,
        subsample=0.9, subsample_freq=1, reg_lambda=1.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    clf_full.fit(x_full, (train_df.attack_cat != "Normal").astype(int),
                 categorical_feature=cat_idx,
                 sample_weight=compute_sample_weight(
                     "balanced", (train_df.attack_cat != "Normal").astype(int)
                 ))
    p_test = clf_full.predict_proba(x_test)[:, list(clf_full.classes_).index(1)]
    clf_flag_test = p_test >= clf_threshold

    validation = {"supervised": evaluate(y_val, clf_flag_val, p_val)}
    test = {"supervised": evaluate(y_test, clf_flag_test, p_test)}
    for name, threshold in anomaly_thresholds.items():
        anomaly_val_flag = anomaly_val >= threshold
        anomaly_test_flag = anomaly_test >= threshold
        union_val = clf_flag_val | anomaly_val_flag
        union_test = clf_flag_test | anomaly_test_flag
        validation[f"isolation_forest_{name}"] = evaluate(y_val, anomaly_val_flag, anomaly_val)
        validation[f"union_{name}"] = evaluate(y_val, union_val)
        test[f"isolation_forest_{name}"] = evaluate(y_test, anomaly_test_flag, anomaly_test)
        test[f"union_{name}"] = evaluate(y_test, union_test)

    report = {
        "dataset": "UNSW-NB15 official row-count partitions; mirror filenames checked by documented sizes",
        "official_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "protocol": "Isolation Forest fit on benign training rows only; thresholds set to validation-benign 98th/99.5th score quantiles; supervised threshold chosen on validation for >=90% recall; report supervised, unsupervised, and OR union on held-out test",
        "train_rows": len(train_df), "test_rows": len(test_df),
        "supervised_attack_threshold": float(clf_threshold),
        "anomaly_thresholds": anomaly_thresholds,
        "validation": validation,
        "test": test,
        "note": "Isolation Forest is an unsupervised proxy, not the app's synthetic-trained Autoencoder. Separate dataset schema; UNSW commercial use requires author permission.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("test metrics")
    for name, m in test.items():
        print(name, {k: round(m[k], 4) for k in
              ["accuracy", "macro_f1", "balanced_accuracy", "attack_precision",
               "attack_recall", "attack_f1", "benign_false_positive_rate"]})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
