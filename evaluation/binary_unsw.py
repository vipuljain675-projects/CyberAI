#!/usr/bin/env python3
"""Binary benign/attack threshold experiment on UNSW-NB15."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from imbalance_unsw import DATA, SEED, load_frame, make_encoder

OUT = Path(__file__).resolve().parent / "reports" / "unsw_nb15_binary_experiment.json"


def choose_threshold(y, score, policy):
    candidates = np.unique(score)
    feasible = []
    for threshold in candidates:
        pred = (score >= threshold).astype(int)
        recall = float(np.mean(pred[y == 1]))
        fpr = float(np.mean(pred[y == 0]))
        accuracy = float(accuracy_score(y, pred))
        if recall >= 0.90:
            feasible.append((accuracy, -fpr, threshold, recall, fpr))
    if policy == "max_accuracy_recall90":
        if not feasible:
            return float(np.min(candidates))
        return float(max(feasible)[2])
    if policy == "min_fpr_recall90":
        if not feasible:
            return float(np.min(candidates))
        return float(min((x[4], -x[2], x[0], x[2]) for x in feasible)[3])
    if policy in {"fpr_cap_2pct", "fpr_cap_0_5pct"}:
        cap = 0.02 if policy == "fpr_cap_2pct" else 0.005
        valid = []
        for threshold in candidates:
            pred = (score >= threshold).astype(int)
            recall = float(np.mean(pred[y == 1]))
            fpr = float(np.mean(pred[y == 0]))
            accuracy = float(accuracy_score(y, pred))
            if fpr <= cap:
                valid.append((recall, threshold, accuracy))
        if valid:
            return float(max(valid)[1])
        return float(np.max(candidates))
    raise ValueError(policy)


def score_metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    p, r, f, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "attack_precision": float(p[1]), "attack_recall": float(r[1]),
        "attack_f1": float(f[1]),
        "benign_false_positive_rate": float(np.mean(pred[y == 0])),
        "attack_roc_auc": float(roc_auc_score(y, score)),
        "attack_average_precision": float(average_precision_score(y, score)),
        "per_class": {
            "benign": {"precision": float(p[0]), "recall": float(r[0]),
                       "f1": float(f[0]), "support": int(support[0])},
            "attack": {"precision": float(p[1]), "recall": float(r[1]),
                       "f1": float(f[1]), "support": int(support[1])},
        },
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
        raise ValueError("UNSW partitions do not have the official 175341/82332 row counts")
    y_all = (train_df.attack_cat != "Normal").astype(int).to_numpy()
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SEED, stratify=y_all
    )
    x_train, x_val, cat_idx, _ = make_encoder(train_df.iloc[train_idx], train_df.iloc[val_idx])
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    x_full, x_test, cat_idx, _ = make_encoder(train_df, test_df)
    y_test = (test_df.attack_cat != "Normal").astype(int).to_numpy()

    candidates = {}
    for mode in ["natural", "balanced"]:
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=350, learning_rate=0.05,
            num_leaves=31, min_child_samples=35, colsample_bytree=0.9,
            subsample=0.9, subsample_freq=1, reg_lambda=1.0,
            random_state=SEED, n_jobs=2, verbosity=-1,
        )
        fit_args = {"categorical_feature": cat_idx}
        if mode == "balanced":
            fit_args["sample_weight"] = compute_sample_weight("balanced", y_train)
        model.fit(x_train, y_train, **fit_args)
        val_score = model.predict_proba(x_val)[:, list(model.classes_).index(1)]
        thresholds = {
            policy: choose_threshold(y_val, val_score, policy)
            for policy in ["max_accuracy_recall90", "min_fpr_recall90",
                           "fpr_cap_2pct", "fpr_cap_0_5pct"]
        }
        candidates[mode] = {
            "validation": {
                policy: score_metrics(y_val, val_score, threshold)
                for policy, threshold in thresholds.items()
            },
            "thresholds": thresholds,
        }
        print(mode, {policy: {k: round(v[k], 4) for k in
              ["accuracy", "attack_precision", "attack_recall", "benign_false_positive_rate"]}
              for policy, v in candidates[mode]["validation"].items()})

    selected = max(candidates, key=lambda m: (
        candidates[m]["validation"]["fpr_cap_2pct"]["attack_recall"],
        candidates[m]["validation"]["fpr_cap_2pct"]["accuracy"],
    ))
    threshold = candidates[selected]["thresholds"]["fpr_cap_2pct"]
    final = lgb.LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, min_child_samples=35, colsample_bytree=0.9,
        subsample=0.9, subsample_freq=1, reg_lambda=1.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    fit_args = {"categorical_feature": cat_idx}
    if selected == "balanced":
        fit_args["sample_weight"] = compute_sample_weight("balanced", y_all)
    final.fit(x_full, y_all, **fit_args)
    test_score = final.predict_proba(x_test)[:, list(final.classes_).index(1)]
    test_results = {
        policy: score_metrics(y_test, test_score, candidates[selected]["thresholds"][policy])
        for policy in ["max_accuracy_recall90", "min_fpr_recall90",
                       "fpr_cap_2pct", "fpr_cap_0_5pct"]
    }
    report = {
        "dataset": "UNSW-NB15 official predefined row counts; public mirror filenames checked by counts",
        "official_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "protocol": "compare unweighted and balanced binary LightGBM on training-only stratified validation; choose model by validation attack recall under FPR <=2%; calibrate thresholds using validation only; score held-out partition",
        "train_rows": len(train_df), "test_rows": len(test_df),
        "selected_model": selected,
        "validation_candidates": candidates,
        "test": test_results,
        "test_majority_baseline_accuracy": float(max(np.mean(y_test == 0), np.mean(y_test == 1))),
        "note": "Separate UNSW binary benchmark; feature schema differs from CyberAI's serving model. Academic-use dataset; commercial use requires author permission.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("selected", selected, "test", {
        policy: {k: round(v[k], 4) for k in
                 ["accuracy", "macro_f1", "balanced_accuracy", "attack_precision",
                  "attack_recall", "attack_f1", "benign_false_positive_rate"]}
        for policy, v in test_results.items()
    })
    print("wrote", OUT)


if __name__ == "__main__":
    main()
