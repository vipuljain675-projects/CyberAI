#!/usr/bin/env python3
"""Validation-calibrated one-vs-rest thresholds for multiclass UNSW-NB15."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from imbalance_unsw import (
    DATA, LABELS, SEED, calibrate_attack_threshold, classifier, load_frame,
    make_encoder, metrics, ordered_probabilities, resample,
)

OUT = Path(__file__).resolve().parent / "reports" / "unsw_nb15_per_class_thresholds.json"


def fit_thresholds(y, prob, labels):
    thresholds = {}
    for i, label in enumerate(labels):
        binary_y = (np.asarray(y) == label).astype(int)
        precision, recall, candidate = precision_recall_curve(binary_y, prob[:, i])
        if len(candidate) == 0:
            thresholds[label] = 1.0
            continue
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        thresholds[label] = float(candidate[int(np.argmax(f1))])
    return thresholds


def predict_thresholded(prob, labels, thresholds, mode):
    t = np.array([thresholds[x] for x in labels])
    eligible = prob >= t[None, :]
    scores = (prob / np.maximum(t[None, :], 1e-12) if mode == "ratio"
              else prob - t[None, :])
    pred = np.full(len(prob), "Normal", dtype=object)
    for row in range(len(prob)):
        if eligible[row].any():
            candidate_scores = np.where(eligible[row], scores[row], -np.inf)
            pred[row] = labels[int(np.argmax(candidate_scores))]
    return pred


def main():
    a = load_frame(DATA / "UNSW_NB15_training-set.csv")
    b = load_frame(DATA / "UNSW_NB15_testing-set.csv")
    if len(a) == 175_341:
        train_df, test_df = a, b
    elif len(b) == 175_341:
        train_df, test_df = b, a
    else:
        raise ValueError("Unexpected partition sizes")
    y_all = train_df.attack_cat.to_numpy()
    labels = sorted(name for name in LABELS if name in set(y_all) | set(test_df.attack_cat))
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SEED, stratify=y_all
    )
    x_train, x_val, cat_idx, _ = make_encoder(train_df.iloc[train_idx], train_df.iloc[val_idx])
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    fit_x, fit_y = resample("undersample", x_train, y_train, cat_idx)
    model = classifier()
    model.fit(fit_x, fit_y, categorical_feature=cat_idx)
    val_prob = ordered_probabilities(model, x_val, labels, cat_idx)
    thresholds = fit_thresholds(y_val, val_prob, labels)

    validation = {}
    for mode in ["ratio", "margin"]:
        pred = predict_thresholded(val_prob, labels, thresholds, mode)
        validation[mode] = metrics(y_val, pred, val_prob, labels)
        validation[mode]["accuracy"] = float(accuracy_score(y_val, pred))
        print(mode, {k: round(validation[mode][k], 4) for k in
                     ["accuracy", "macro_f1", "balanced_accuracy", "attack_precision",
                      "attack_recall", "benign_false_positive_rate"]})
    selected = max(validation, key=lambda m: (validation[m]["macro_f1"], validation[m]["accuracy"]))

    full_x, test_x, cat_idx, _ = make_encoder(train_df, test_df)
    full_y = train_df.attack_cat.to_numpy()
    fit_x, fit_y = resample("undersample", full_x, full_y, cat_idx)
    final = classifier()
    final.fit(fit_x, fit_y, categorical_feature=cat_idx)
    test_prob = ordered_probabilities(final, test_x, labels, cat_idx)
    y_test = test_df.attack_cat.to_numpy()
    pred = predict_thresholded(test_prob, labels, thresholds, selected)
    test_metrics = metrics(y_test, pred, test_prob, labels)
    test_metrics["accuracy"] = float(accuracy_score(y_test, pred))
    report = {
        "dataset": "UNSW-NB15 official row-count partitions; mirror filenames resolved by documented sizes",
        "official_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "protocol": "undersampled LightGBM; per-class one-vs-rest F1 thresholds fit on stratified training-only validation; threshold-combination rule selected on validation macro-F1; final held-out test",
        "train_rows": len(train_df), "test_rows": len(test_df),
        "calibrated_class_thresholds": thresholds,
        "validation": validation,
        "selected_combination_rule": selected,
        "test": test_metrics,
        "note": "Exploratory multiclass threshold study; same held-out partition has been used in related model experiments. It is not an untouched confirmatory evaluation or a model artifact for CyberAI's 91-feature serving schema.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("selected", selected, "test", {k: round(test_metrics[k], 4) for k in
          ["accuracy", "macro_f1", "balanced_accuracy", "macro_ovr_roc_auc",
           "attack_precision", "attack_recall", "benign_false_positive_rate"]})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
