#!/usr/bin/env python3
"""Compare leakage-safe imbalance strategies on the UNSW-NB15 official split.

This is a separate research benchmark. UNSW-NB15's feature schema is not the
same as CyberAI's serving adapter, so the resulting model is not a drop-in
replacement for the application artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "unsw_nb15"
REPORT = ROOT / "evaluation" / "reports" / "unsw_nb15_imbalance_experiment.json"
SEED = 42
LABELS = ["Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
          "Generic", "Reconnaissance", "Shellcode", "Worms"]


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"attack_cat", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path.name} missing target columns: {required - set(df.columns)}")
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    return df


def make_encoder(train: pd.DataFrame, test: pd.DataFrame):
    drop_cols = {"attack_cat", "label", "id"}
    cols = [c for c in train.columns if c not in drop_cols and c in test.columns]
    cats = [c for c in cols if not pd.api.types.is_numeric_dtype(train[c])]
    nums = [c for c in cols if c not in cats]

    train_c = train[cats].fillna("__MISSING__").astype(str)
    test_c = test[cats].fillna("__MISSING__").astype(str)
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             dtype=np.float64)
    train_cat = ordinal.fit_transform(train_c) if cats else np.empty((len(train), 0))
    test_cat = ordinal.transform(test_c) if cats else np.empty((len(test), 0))

    def numeric(frame):
        vals = frame[nums].apply(pd.to_numeric, errors="coerce")
        vals = vals.replace([np.inf, -np.inf], np.nan)
        return vals.to_numpy(dtype=np.float64)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_num = imputer.fit_transform(numeric(train)) if nums else np.empty((len(train), 0))
    test_num = imputer.transform(numeric(test)) if nums else np.empty((len(test), 0))

    # Keep all numeric columns first and mark ordinal columns as categorical.
    names = nums + cats
    train_x = pd.DataFrame(np.column_stack([train_num, train_cat]), columns=names)
    test_x = pd.DataFrame(np.column_stack([test_num, test_cat]), columns=names)
    cat_idx = list(range(len(nums), len(names)))
    categories = {}
    for j, col in enumerate(cats):
        levels = list(range(len(ordinal.categories_[j])))
        dtype = pd.CategoricalDtype(categories=levels)
        train_x[col] = train_x[col].astype(int).astype(dtype)
        test_x[col] = test_x[col].where(test_x[col].isin(levels), np.nan).astype(dtype)
        categories[col] = levels
    return train_x, test_x, cat_idx, categories


def classifier():
    return lgb.LGBMClassifier(
        objective="multiclass", n_estimators=350, learning_rate=0.05,
        num_leaves=31, max_depth=-1, min_child_samples=35,
        colsample_bytree=0.9, subsample=0.9, subsample_freq=1,
        reg_lambda=1.0, random_state=SEED, n_jobs=2, verbosity=-1,
    )


def forest_frame(x, cat_idx):
    out = x.copy()
    for i in cat_idx:
        col = out.columns[i]
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].cat.codes
    return out


def fit_candidate(mode, x, y, cat_idx):
    if mode == "balanced_random_forest":
        model = BalancedRandomForestClassifier(
            n_estimators=200, max_depth=18, min_samples_leaf=2,
            max_features="sqrt", sampling_strategy="all",
            replacement=False, bootstrap=False, n_jobs=2,
            random_state=SEED,
        )
        model.fit(forest_frame(x, cat_idx), y)
        return model

    x_fit, y_fit = resample(mode, x, y, cat_idx)
    model = classifier()
    fit_args = {"categorical_feature": cat_idx}
    if mode == "class_weight":
        fit_args["sample_weight"] = compute_sample_weight("balanced", y_fit)
    model.fit(x_fit, y_fit, **fit_args)
    return model


def resample(mode, x, y, cat_idx, seed=SEED):
    if mode == "natural":
        return x, y
    if mode == "class_weight":
        return x, y
    if mode == "undersample":
        counts = pd.Series(y).value_counts()
        # Cap each majority class at 20,000 while leaving very rare classes intact.
        target = {cls: min(int(count), 20_000) for cls, count in counts.items()}
        sampler = RandomUnderSampler(sampling_strategy=target, random_state=seed)
    elif mode == "smotenc":
        min_count = int(pd.Series(y).value_counts().min())
        if min_count < 2:
            raise ValueError("SMOTENC needs at least two samples per class")
        sampler = SMOTENC(
            categorical_features=cat_idx,
            sampling_strategy="not majority",
            k_neighbors=min(5, min_count - 1),
            random_state=seed,
        )
    else:
        raise ValueError(mode)
    sampled_x, sampled_y = sampler.fit_resample(x, y)
    sampled_x = pd.DataFrame(sampled_x, columns=x.columns)
    for col in x.columns:
        if isinstance(x[col].dtype, pd.CategoricalDtype):
            sampled_x[col] = sampled_x[col].round().astype(int).astype(x[col].dtype)
    return sampled_x, np.asarray(sampled_y)


def metrics(y_true, pred, prob, labels):
    p, r, f, support = precision_recall_fscore_support(
        y_true, pred, labels=labels, zero_division=0
    )
    attack_true = (np.asarray(y_true) != "Normal").astype(int)
    attack_pred = (np.asarray(pred) != "Normal").astype(int)
    attack_p, attack_r, attack_f, _ = precision_recall_fscore_support(
        attack_true, attack_pred, average="binary", zero_division=0
    )
    benign = np.asarray(y_true) == "Normal"
    fpr = float(np.mean(np.asarray(pred)[benign] != "Normal")) if benign.any() else None
    per_class = {}
    pr_curves = {}
    recall_grid = np.linspace(0.0, 1.0, 101)
    y_arr = np.asarray(y_true)
    for i, label in enumerate(labels):
        binary_y = (y_arr == label).astype(int)
        precision_curve, recall_curve, _ = precision_recall_curve(binary_y, prob[:, i])
        per_class[label] = {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f[i]), "support": int(support[i]),
            "average_precision": float(average_precision_score(binary_y, prob[:, i])),
        }
        pr_curves[label] = {
            "recall": recall_grid.tolist(),
            "precision": np.interp(
                recall_grid, recall_curve[::-1], precision_curve[::-1]
            ).tolist(),
        }
    attack_score = 1.0 - prob[:, labels.index("Normal")]
    attack_precision_curve, attack_recall_curve, _ = precision_recall_curve(attack_true, attack_score)
    attack_pr = {
        "recall": recall_grid.tolist(),
        "precision": np.interp(
            recall_grid, attack_recall_curve[::-1], attack_precision_curve[::-1]
        ).tolist(),
    }
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "attack_precision": float(attack_p),
        "attack_recall": float(attack_r),
        "attack_f1": float(attack_f),
        "attack_average_precision": float(average_precision_score(attack_true, attack_score)),
        "benign_false_positive_rate": fpr,
        "per_class": per_class,
        "per_class_pr_curves": pr_curves,
        "attack_pr_curve": attack_pr,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }
    try:
        out["macro_ovr_roc_auc"] = float(roc_auc_score(
            y_true, prob, labels=labels, multi_class="ovr", average="macro"
        ))
    except ValueError as exc:
        out["macro_ovr_roc_auc"] = None
        out["roc_auc_note"] = str(exc)
    return out


def ordered_probabilities(model, x, labels, cat_idx):
    if isinstance(model, BalancedRandomForestClassifier):
        x = forest_frame(x, cat_idx)
    raw = model.predict_proba(x)
    class_order = list(model.classes_)
    return raw[:, [class_order.index(label) for label in labels]]


def calibrate_attack_threshold(y_true, prob, labels, min_recall=0.90):
    """Choose the most conservative attack threshold meeting validation recall."""
    normal_idx = labels.index("Normal")
    attack_score = 1.0 - prob[:, normal_idx]
    truth_attack = np.asarray(y_true) != "Normal"
    candidates = np.unique(attack_score)
    valid = []
    for threshold in candidates:
        flagged = attack_score >= threshold
        recall = float(np.mean(flagged[truth_attack])) if truth_attack.any() else 0.0
        fpr = float(np.mean(flagged[~truth_attack])) if (~truth_attack).any() else 0.0
        if recall >= min_recall:
            valid.append((fpr, -threshold, float(threshold), recall))
    if not valid:
        return 0.0
    return min(valid)[2]


def threshold_predictions(prob, labels, threshold):
    normal_idx = labels.index("Normal")
    attack_indices = [i for i, name in enumerate(labels) if name != "Normal"]
    pred = np.empty(len(prob), dtype=object)
    flagged = (1.0 - prob[:, normal_idx]) >= threshold
    pred[~flagged] = "Normal"
    if flagged.any():
        attack_pred = np.asarray(labels, dtype=object)[attack_indices][
            np.argmax(prob[flagged][:, attack_indices], axis=1)
        ]
        pred[flagged] = attack_pred
    return pred


def main():
    named_train = load_frame(DATA / "UNSW_NB15_training-set.csv")
    named_test = load_frame(DATA / "UNSW_NB15_testing-set.csv")
    # The public mirror reverses these two filenames: the official UNSW page
    # specifies 175,341 training rows and 82,332 testing rows.
    if len(named_train) == 175_341 and len(named_test) == 82_332:
        train_df, test_df = named_train, named_test
        partition_note = "mirror filenames match official row counts"
    elif len(named_train) == 82_332 and len(named_test) == 175_341:
        train_df, test_df = named_test, named_train
        partition_note = "mirror filenames reversed; assigned train/test by official row counts"
    else:
        raise ValueError(
            f"Unexpected partition sizes: named train={len(named_train)}, "
            f"named test={len(named_test)}; expected 175341/82332"
        )
    y_all = train_df["attack_cat"].to_numpy()
    labels = sorted(name for name in LABELS if name in set(y_all) | set(test_df["attack_cat"]))
    train_part, val_part = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=SEED,
        stratify=y_all,
    )
    train_x, val_x, cat_idx, categories = make_encoder(
        train_df.iloc[train_part], train_df.iloc[val_part]
    )
    y_train, y_val = y_all[train_part], y_all[val_part]

    # Compare on validation groups only. The provided test partition is held back
    # until the best strategy has been selected by validation macro-F1.
    validation = {}
    # SMOTENC was already tested once; omit it from repeat runs because its
    # temporary resampled matrix can exceed this host's memory cap.
    strategies = ["natural", "class_weight", "undersample", "balanced_random_forest"]
    for mode in strategies:
        model = fit_candidate(mode, train_x, y_train, cat_idx)
        probs = ordered_probabilities(model, val_x, labels, cat_idx)
        threshold = calibrate_attack_threshold(y_val, probs, labels, min_recall=0.90)
        p = threshold_predictions(probs, labels, threshold)
        validation[mode] = metrics(y_val, p, probs, labels)
        validation[mode]["attack_threshold"] = threshold
        validation[mode]["argmax_accuracy"] = float(accuracy_score(y_val, model.predict(val_x)))
        print(mode, {k: round(validation[mode][k], 4) for k in
                     ["accuracy", "macro_f1", "balanced_accuracy", "attack_recall",
                      "attack_precision", "benign_false_positive_rate"]})

    selected = max(validation, key=lambda k: (validation[k]["macro_f1"], validation[k]["accuracy"]))
    full_x, test_x, cat_idx, categories = make_encoder(train_df, test_df)
    full_y, y_test = train_df["attack_cat"].to_numpy(), test_df["attack_cat"].to_numpy()
    final = fit_candidate(selected, full_x, full_y, cat_idx)
    final_prob = ordered_probabilities(final, test_x, labels, cat_idx)
    selected_threshold = validation[selected]["attack_threshold"]
    final_pred = threshold_predictions(final_prob, labels, selected_threshold)
    test_metrics = metrics(y_test, final_pred, final_prob, labels)
    test_metrics["attack_threshold"] = selected_threshold

    report = {
        "dataset": "UNSW-NB15 official predefined training/testing partitions",
        "mirror": "https://github.com/tichill/UNSW_DATA",
        "official_source": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "protocol": "resampling/model strategy selected by macro-F1 on a stratified 20% validation split of training partition; binary attack threshold calibrated on validation for >=90% attack recall; one final evaluation on the predefined test partition",
        "partition_file_note": partition_note,
        "train_rows": len(train_df), "test_rows": len(test_df),
        "train_class_counts": train_df["attack_cat"].value_counts().to_dict(),
        "test_class_counts": test_df["attack_cat"].value_counts().to_dict(),
        "features": list(full_x.columns),
        "categorical_features": [full_x.columns[i] for i in cat_idx],
        "validation": validation,
        "prior_smotenc_validation": {
            "accuracy": 0.7802, "macro_f1": 0.5861,
            "balanced_accuracy": 0.6142, "attack_precision": 0.9918,
            "attack_recall": 0.9000, "benign_false_positive_rate": 0.0158,
            "attack_threshold": 0.8775,
            "note": "Recorded from the earlier fold-local SMOTENC-only validation run; no final-test score was produced for that model.",
        },
        "selected_strategy": selected,
        "test": test_metrics,
        "note": "Separate research benchmark; schema differs from the 91-feature serving adapter. UNSW permits academic use; commercial use requires permission from dataset authors.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("selected strategy:", selected)
    print("held-out test:", json.dumps({k: v for k, v in test_metrics.items() if k != "per_class" and k != "confusion_matrix"}, indent=2))
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
