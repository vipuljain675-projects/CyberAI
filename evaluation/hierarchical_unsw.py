#!/usr/bin/env python3
"""Validation-selected binary detector + attack-family classifier on UNSW-NB15."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from imbalance_unsw import (
    DATA, LABELS, REPORT, SEED, calibrate_attack_threshold, load_frame,
    make_encoder, metrics, ordered_probabilities, threshold_predictions,
)

OUT = REPORT.with_name("unsw_nb15_hierarchical_experiment.json")


def binary_model():
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=350, learning_rate=0.05,
        num_leaves=31, min_child_samples=35, colsample_bytree=0.9,
        subsample=0.9, subsample_freq=1, reg_lambda=1.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )


def attack_model():
    return lgb.LGBMClassifier(
        objective="multiclass", n_estimators=350, learning_rate=0.05,
        num_leaves=31, min_child_samples=25, colsample_bytree=0.9,
        subsample=0.9, subsample_freq=1, reg_lambda=1.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )


def fit_attack_model(x, y, cat_idx, weighting):
    model = attack_model()
    fit_args = {"categorical_feature": cat_idx}
    if weighting == "balanced":
        fit_args["sample_weight"] = compute_sample_weight("balanced", y)
    elif weighting == "sqrt_weight":
        weights = np.sqrt(compute_sample_weight("balanced", y))
        fit_args["sample_weight"] = weights / weights.mean()
    model.fit(x, y, **fit_args)
    return model


def joint_probability(p_attack, p_subtype, subtype_classes, labels):
    out = np.zeros((len(p_attack), len(labels)), dtype=np.float64)
    normal_idx = labels.index("Normal")
    out[:, normal_idx] = 1.0 - p_attack
    for j, cls in enumerate(subtype_classes):
        out[:, labels.index(cls)] = p_attack * p_subtype[:, j]
    return out


def main():
    named_train = load_frame(DATA / "UNSW_NB15_training-set.csv")
    named_test = load_frame(DATA / "UNSW_NB15_testing-set.csv")
    if len(named_train) == 175_341 and len(named_test) == 82_332:
        train_df, test_df = named_train, named_test
        partition_note = "mirror filenames match official row counts"
    elif len(named_train) == 82_332 and len(named_test) == 175_341:
        train_df, test_df = named_test, named_train
        partition_note = "mirror filenames reversed; assigned by official row counts"
    else:
        raise ValueError(f"Unexpected UNSW partition sizes: {len(named_train)}, {len(named_test)}")

    y_all = train_df["attack_cat"].to_numpy()
    labels = sorted(name for name in LABELS if name in set(y_all) | set(test_df["attack_cat"]))
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SEED, stratify=y_all
    )
    x_train, x_val, cat_idx, _ = make_encoder(
        train_df.iloc[train_idx], train_df.iloc[val_idx]
    )
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    is_attack = y_train != "Normal"
    y_attack = y_train[is_attack]

    detector = binary_model()
    detector.fit(x_train, is_attack.astype(int), categorical_feature=cat_idx)
    d_order = list(detector.classes_)
    p_attack_val = detector.predict_proba(x_val)[:, d_order.index(1)]

    validation = {}
    subtype_modes = ["natural", "balanced", "sqrt_weight"]
    for mode in subtype_modes:
        subtype = fit_attack_model(x_train.loc[is_attack], y_attack, cat_idx, mode)
        subtype_classes = list(subtype.classes_)
        subtype_p = ordered_probabilities(subtype, x_val, subtype_classes, cat_idx)
        joint_p = joint_probability(p_attack_val, subtype_p, subtype_classes, labels)
        threshold = calibrate_attack_threshold(y_val, joint_p, labels, min_recall=0.90)
        pred = threshold_predictions(joint_p, labels, threshold)
        validation[mode] = metrics(y_val, pred, joint_p, labels)
        validation[mode]["attack_threshold"] = threshold
        print(mode, {k: round(validation[mode][k], 4) for k in
                     ["accuracy", "macro_f1", "balanced_accuracy", "attack_precision",
                      "attack_recall", "benign_false_positive_rate"]})

    selected = max(validation, key=lambda k: (validation[k]["macro_f1"], validation[k]["accuracy"]))

    full_x, test_x, cat_idx, _ = make_encoder(train_df, test_df)
    full_y = train_df["attack_cat"].to_numpy()
    full_is_attack = full_y != "Normal"
    final_detector = binary_model()
    final_detector.fit(full_x, full_is_attack.astype(int), categorical_feature=cat_idx)
    detector_order = list(final_detector.classes_)
    p_attack_test = final_detector.predict_proba(test_x)[:, detector_order.index(1)]

    final_subtype = fit_attack_model(
        full_x.loc[full_is_attack], full_y[full_is_attack], cat_idx, selected
    )
    subtype_classes = list(final_subtype.classes_)
    p_subtype_test = ordered_probabilities(final_subtype, test_x, subtype_classes, cat_idx)
    test_probability = joint_probability(p_attack_test, p_subtype_test, subtype_classes, labels)
    threshold = validation[selected]["attack_threshold"]
    test_prediction = threshold_predictions(test_probability, labels, threshold)
    y_test = test_df["attack_cat"].to_numpy()
    test_metrics = metrics(y_test, test_prediction, test_probability, labels)
    test_metrics["attack_threshold"] = threshold

    report = {
        "dataset": "UNSW-NB15 partitions assigned by the official documented row counts",
        "mirror": "https://github.com/tichill/UNSW_DATA",
        "official_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "partition_file_note": partition_note,
        "protocol": "binary detector trained on training rows; attack subtype model trained on training attacks; subtype weighting selected by macro-F1 on a stratified training-only validation split; attack threshold calibrated on validation for >=90% recall; one held-out test evaluation",
        "train_rows": len(train_df), "test_rows": len(test_df),
        "selected_subtype_weighting": selected,
        "validation": validation,
        "test": test_metrics,
        "note": "Separate research benchmark; feature schema differs from CyberAI's 91-feature serving model. UNSW allows academic use; obtain author permission for commercial use.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print("selected subtype weighting:", selected)
    print("test metrics:", {k: round(v, 4) for k, v in test_metrics.items()
                             if isinstance(v, (int, float))})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
