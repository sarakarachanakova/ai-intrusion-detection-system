"""
Тренирање на истите четири модели (Decision Tree, Random Forest, Logistic
Regression, MLP) врз множеството податоци UNSW-NB15, со ист pipeline
(ColumnTransformer: StandardScaler + OneHotEncoder) како и за NSL-KDD.

Ова е директно продолжение на делот 8.5 "Обука и тестирање со понови
множества податоци" од трудот — иста методологија, применета врз понов
датасет, со официјалната предефинирана поделба training-set/testing-set.
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS_ROOT = os.path.join(BASE, "models")

CAT_COLS = ["proto", "service", "state"]
DROP_COLS = ["attack_cat", "label"]
MODES = ["binary", "multiclass"]


def load_data():
    df_train = pd.read_parquet(os.path.join(DATA, "UNSW_NB15_training-set.parquet"))
    df_test = pd.read_parquet(os.path.join(DATA, "UNSW_NB15_testing-set.parquet"))
    return df_train, df_test


def build_preprocessor(num_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


def build_models():
    return {
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "mlp_neural_network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
    }


def make_xy(df, mode, feature_cols):
    X = df[feature_cols].copy()
    if mode == "binary":
        y = df["label"].astype(int)
    else:
        y = df["attack_cat"].astype(str)
    return X, y


def main():
    df_train, df_test = load_data()
    feature_cols = [c for c in df_train.columns if c not in DROP_COLS]
    num_cols = [c for c in feature_cols if c not in CAT_COLS]

    print("Train:", df_train.shape, " Test:", df_test.shape)
    print("Feature columns:", len(feature_cols), " (num:", len(num_cols), ", cat:", len(CAT_COLS), ")")

    summary_rows = []
    for mode in MODES:
        out_dir = os.path.join(MODELS_ROOT, mode)
        os.makedirs(out_dir, exist_ok=True)
        X_tr, y_tr = make_xy(df_train, mode, feature_cols)
        X_te, y_te = make_xy(df_test, mode, feature_cols)
        print("\n" + "=" * 70 + "\nMODE:", mode, "\n" + "=" * 70)

        for key, clf in build_models().items():
            pipe = Pipeline([
                ("preprocessor", build_preprocessor(num_cols)),
                ("clf", clf),
            ])
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            secs = time.time() - t0
            y_pred = pipe.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            f1 = (f1_score(y_te, y_pred, pos_label=1, zero_division=0) if mode == "binary"
                  else f1_score(y_te, y_pred, average="macro", zero_division=0))

            path = os.path.join(out_dir, key + ".joblib")
            # compress=3: proto has 133 categories -> deep RF trees are huge uncompressed
            joblib.dump(pipe, path, compress=3)

            rep = classification_report(y_te, y_pred, digits=4, zero_division=0)
            with open(os.path.join(out_dir, key + "_report.txt"), "w") as f:
                f.write("MODE=%s  MODEL=%s  DATASET=UNSW-NB15\nAccuracy=%.4f  F1=%.4f  train_time=%.1fs\n\n%s"
                        % (mode, key, acc, f1, secs, rep))

            summary_rows.append({"mode": mode, "model": key,
                                  "accuracy": round(acc, 4), "f1": round(f1, 4),
                                  "train_s": round(secs, 1)})
            print("  %-22s acc=%.4f  f1=%.4f  (%.1fs)  -> %s" % (key, acc, f1, secs, path))

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== РЕЗИМЕ ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(MODELS_ROOT, "summary.csv"), index=False)

    metadata = {
        "dataset": "UNSW-NB15 (UNSW_NB15_training-set / UNSW_NB15_testing-set)",
        "modes": MODES,
        "models": list(build_models().keys()),
        "label_column": "label",
        "multiclass_column": "attack_cat",
        "feature_columns": feature_cols,
        "categorical_columns": CAT_COLS,
        "numeric_columns": num_cols,
        "binary_mapping": {"0": "normal", "1": "attack"},
        "attack_categories": sorted(df_train["attack_cat"].astype(str).unique().tolist()),
    }
    with open(os.path.join(MODELS_ROOT, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nГотово. Модели, извештаи и метаподатоци се зачувани во", MODELS_ROOT)


if __name__ == "__main__":
    main()
