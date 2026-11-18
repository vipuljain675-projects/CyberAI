#!/usr/bin/env python3
"""Plot sampled test-set precision-recall curves from the UNSW report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "evaluation" / "reports" / "unsw_nb15_imbalance_experiment.json"
OUT = REPORT.with_name("unsw_nb15_pr_curves.png")

report = json.loads(REPORT.read_text())
curves = report["test"]["per_class_pr_curves"]
class_metrics = report["test"]["per_class"]
attack = report["test"]["attack_pr_curve"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for name, curve in curves.items():
    ap = class_metrics[name]["average_precision"]
    axes[0].plot(curve["recall"], curve["precision"], label=f"{name} (AP={ap:.3f})")
axes[0].set_title("UNSW-NB15 held-out test: attack family PR")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1.02)
axes[0].grid(alpha=0.25)
axes[0].legend(fontsize=8, loc="lower left")

axes[1].plot(attack["recall"], attack["precision"], color="#c0392b", linewidth=2)
axes[1].set_title(
    f"Binary attack PR (AP={report['test']['attack_average_precision']:.3f})"
)
axes[1].set_xlabel("Attack recall")
axes[1].set_ylabel("Attack precision")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1.02)
axes[1].grid(alpha=0.25)

fig.suptitle(
    f"UNSW-NB15 exploratory test PR curves — {report['selected_strategy']}\n"
    "Separate research dataset; not CyberAI capture test",
    fontsize=12,
)
fig.savefig(OUT, dpi=160)
print(OUT)
