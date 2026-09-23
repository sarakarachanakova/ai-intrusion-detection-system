import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve,
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
OUT = "/Users/sara/Desktop/master/1. Digitalna transformacija I VI/dtvi-project/paper/figures_unsw"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"

df_test = pd.read_parquet(os.path.join(DATA, "UNSW_NB15_testing-set.parquet"))
y_bin = df_test["label"].astype(int)
y_multi = df_test["attack_cat"].astype(str)

mk_names = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "mlp_neural_network": "MLP Neural Network",
}

binary_models = {k: joblib.load(f"{MODELS}/binary/{k}.joblib") for k in mk_names}

# ---- class distribution (multiclass, test set) ----
order = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
counts = y_multi.value_counts().reindex(order)
plt.figure(figsize=(9, 5))
bars = plt.bar(order, counts.values, color=plt.cm.tab10.colors)
for b, v in zip(bars, counts.values):
    plt.text(b.get_x() + b.get_width() / 2, v + 200, str(int(v)), ha="center", fontsize=9)
plt.xticks(rotation=30, ha="right")
plt.ylabel("Број на примероци")
plt.title("Распределба на класите во тест-множеството (UNSW-NB15)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_class_distribution_unsw.png", dpi=150)
plt.close()

# ---- ROC curves (binary) ----
plt.figure(figsize=(7, 6))
auc_values = {}
for key, pipe in binary_models.items():
    prob = pipe.predict_proba(df_test.drop(columns=["attack_cat", "label"]))[:, list(pipe.classes_).index(1)]
    fpr, tpr, _ = roc_curve(y_bin, prob)
    auc_val = auc(fpr, tpr)
    auc_values[key] = round(float(auc_val), 4)
    plt.plot(fpr, tpr, label=f"{mk_names[key]} (AUC={auc_val:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Случајно погодување (AUC=0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-криви за бинарната класификација (UNSW-NB15)")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_roc_curves_unsw.png", dpi=150)
plt.close()
print("AUC values:", auc_values)
with open(f"{MODELS}/auc_binary.json", "w") as f:
    json.dump(auc_values, f, indent=2)

# ---- binary confusion matrices (Macedonian labels) ----
labels_bin = ["нормален", "напад"]
for key, pipe in binary_models.items():
    y_pred = pipe.predict(df_test.drop(columns=["attack_cat", "label"]))
    cm = confusion_matrix(y_bin, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=labels_bin).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Матрица на конфузија — {mk_names[key]} (UNSW-NB15)")
    ax.set_xlabel("Предвидена класа")
    ax.set_ylabel("Вистинска класа")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_cm_{key}_unsw.png", dpi=150)
    plt.close()

# ---- multiclass confusion matrix (RF) ----
rf_multi = joblib.load(f"{MODELS}/multiclass/random_forest.joblib")
y_pred_multi = rf_multi.predict(df_test.drop(columns=["attack_cat", "label"]))
cm = confusion_matrix(y_multi, y_pred_multi, labels=order)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=order).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d", xticks_rotation=45)
ax.set_title("Матрица на конфузија — Random Forest, многукласно (UNSW-NB15)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_cm_multiclass_rf_unsw.png", dpi=150)
plt.close()

# ---- feature importance (RF binary) ----
rf_pipe = binary_models["random_forest"]
ohe = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"]
num_cols = [c for c in df_test.columns if c not in ["attack_cat", "label", "proto", "service", "state"]]
feat_names = num_cols + list(ohe.get_feature_names_out(["proto", "service", "state"]))
importances = rf_pipe.named_steps["clf"].feature_importances_
order_idx = np.argsort(importances)[::-1][:20]
plt.figure(figsize=(9, 7))
plt.barh(np.array(feat_names)[order_idx][::-1], importances[order_idx][::-1], color="#4C72B0")
plt.xlabel("Важност (Gini importance)")
plt.title("Топ 20 најважни карактеристики (Random Forest, бинарно, UNSW-NB15)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_feature_importance_unsw.png", dpi=150)
plt.close()

print("Done. Figures written to", OUT)
