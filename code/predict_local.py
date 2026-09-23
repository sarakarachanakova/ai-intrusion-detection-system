#!/usr/bin/env python3
"""
Локално тестирање на веќе истрениран IDS модел (NSL-KDD) врз НОВ датасет.
Не тренира ништо — само ги вчитува зачуваните модели и предвидува/евалуира.

Пример:
    python3 predict_local.py --models-dir models --data KDDTest+.txt --mode binary
    python3 predict_local.py --models-dir models --data nov_dataset.csv --mode multiclass --include-keras

Новиот датасет треба да биде во NSL-KDD формат (исти 41 атрибут во ист редослед).
Поддржани се:
  * 43 колони  -> 41 атрибут + label + difficulty
  * 42 колони  -> 41 атрибут + label
  * 41 колони  -> само атрибути (без label; тогаш само се предвидува, без евалуација)
  * CSV со заглавје што ги содржи имињата на колоните
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import joblib

# Истите дефиниции како во тренинг-нотбукот (за читање на нов датасет и мапирање на labels) ---
COL_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

ATTACK_MAP = {
    "normal": "normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "worm": "DoS", "mailbomb": "DoS",
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L", "multihop": "R2L",
    "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L", "xlock": "R2L", "xsnoop": "R2L",
    "snmpguess": "R2L", "snmpgetattack": "R2L", "httptunnel": "R2L", "sendmail": "R2L",
    "named": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

SKLEARN_MODELS = [
    "decision_tree", "random_forest", "logistic_regression", "mlp_neural_network",
]


def load_metadata(models_dir):
    meta_path = os.path.join(models_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print("ПРЕДУПРЕДУВАЊЕ: нема metadata.json во", models_dir)
        return {}
    with open(meta_path) as f:
        return json.load(f)


def check_versions(meta, models_dir):
    """Предупреди ако локалната верзија на scikit-learn се разликува од таа во Colab."""
    try:
        import sklearn
        trained_v = meta.get("versions", {}).get("scikit_learn")
        if trained_v and trained_v != sklearn.__version__:
            print("\n" + "!" * 72)
            print("ВНИМАНИЕ: scikit-learn локално = %s, а моделите се тренирани со %s."
                  % (sklearn.__version__, trained_v))
            print("Ако вчитувањето падне или дава чудни резултати, инсталирај:")
            print("    pip install -r %s" % os.path.join(models_dir, "requirements_local.txt"))
            print("!" * 72 + "\n")
    except Exception:
        pass


def read_new_dataset(path, feature_columns, label_column="label"):
    """Чита нов датасет робусно (со/без заглавје, со/без label/difficulty)."""
    # Дали првиот ред е заглавје?
    with open(path, "r", errors="ignore") as f:
        first = f.readline().strip()
    first_tokens = [t.strip().strip('"') for t in first.split(",")]

    def _is_number(t):
        try:
            float(t)
            return True
        except ValueError:
            return False

    # Вистинско заглавје: содржи повеќе имиња на колони И првото поле НЕ е број.
    # (Секој NSL-KDD ред со податоци почнува со 'duration' = број, па вредност како
    #  нападот 'land' — која случајно е и име на колона — нема да го измами детекторот.)
    n_name_matches = sum(1 for tok in first_tokens if tok in COL_NAMES)
    has_header = n_name_matches >= 5 and not _is_number(first_tokens[0])

    if has_header:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
    else:
        raw = pd.read_csv(path, header=None)
        ncols = raw.shape[1]
        if ncols == len(COL_NAMES):            # 43: features + label + difficulty
            raw.columns = COL_NAMES
            raw = raw.drop(columns=["difficulty"])
        elif ncols == len(COL_NAMES) - 1:      # 42: features + label
            raw.columns = COL_NAMES[:-1]
        elif ncols == len(feature_columns):    # 41: features only
            raw.columns = feature_columns
        else:
            raise ValueError(
                "Неочекуван број колони (%d). Очекувано 41/42/43 за NSL-KDD формат, "
                "или CSV со заглавје со имиња на колоните." % ncols)
        df = raw

    # X: точно атрибутите по редослед од тренингот; недостасувачки -> пополни
    X = pd.DataFrame()
    cat = set(["protocol_type", "service", "flag"])
    missing = []
    for col in feature_columns:
        if col in df.columns:
            X[col] = df[col]
        else:
            missing.append(col)
            X[col] = "missing" if col in cat else 0
    if missing:
        print("ПРЕДУПРЕДУВАЊЕ: недостасуваат атрибути, пополнети со подразбирана вредност:", missing)

    # Ако премалку атрибути се пронајдени, датасетот веројатно не е во NSL-KDD формат.
    n_present = len(feature_columns) - len(missing)
    if n_present < max(1, len(feature_columns) // 2):
        raise ValueError(
            "Датасетот не одговара на NSL-KDD шемата: пронајдени се само %d/%d атрибути. "
            "Провери ги имињата/редоследот на колоните или дали датасетот е во NSL-KDD формат "
            "(41 атрибут + опц. label/difficulty)." % (n_present, len(feature_columns)))

    for col in feature_columns:
        if col not in cat:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    y_raw = df[label_column] if label_column in df.columns else None
    return X, y_raw


def evaluate(name, y_true, y_pred, out_dir):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    print("\n=== %s ===" % name)
    print("Accuracy: %.4f" % acc)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix — " + name)
        plt.tight_layout()
        png = os.path.join(out_dir, "cm_" + name + ".png")
        plt.savefig(png, dpi=120)
        plt.close(fig)
        print("Зачувана confusion matrix ->", png)
    except Exception as e:
        print("(прескокнат график:", e, ")")
    return acc


def map_labels(y_raw, mode):
    if mode == "binary":
        # Нумеричка label колона (пр. 0/1 веќе енкодирано) — претпостави 0=normal.
        if pd.api.types.is_numeric_dtype(y_raw):
            uniq = sorted(set(pd.unique(y_raw.dropna())))
            if not set(uniq) <= {0, 1}:
                print("ПРЕДУПРЕДУВАЊЕ: нумеричка label колона со вредности %s — "
                      "се претпоставува 0=normal, сѐ друго=напад." % uniq)
            return (y_raw != 0).astype(int)
        return (y_raw.astype(str) != "normal").astype(int)
    # Многукласно: предупреди ако новиот датасет има напад што не е во ATTACK_MAP.
    y_str = y_raw.astype(str)
    unknown_mask = ~y_str.isin(ATTACK_MAP)
    if unknown_mask.any():
        bad = sorted(y_str[unknown_mask].unique().tolist())
        print("ПРЕДУПРЕДУВАЊЕ: %d редови имаат label што не е во ATTACK_MAP "
              "(мапирани во 'unknown'; моделот не може да ги предвиди -> ќе се сметаат за погрешни). "
              "Непознати: %s" % (int(unknown_mask.sum()), bad))
    return y_str.map(lambda x: ATTACK_MAP.get(x, "unknown"))


def main():
    ap = argparse.ArgumentParser(description="Локално тестирање на истрениран NSL-KDD IDS модел.")
    ap.add_argument("--models-dir", default="models", help="папка со зачуваните модели (од nsl_kdd_models.zip)")
    ap.add_argument("--data", required=True, help="патека до новиот датасет (NSL-KDD формат)")
    ap.add_argument("--mode", default="binary", choices=["binary", "multiclass"],
                    help="кој сет модели да се користи")
    ap.add_argument("--models", nargs="*", default=None,
                    help="подмножество модели (decision_tree random_forest ...). По default сите.")
    ap.add_argument("--include-keras", action="store_true",
                    help="тестирај ја и Keras невронската мрежа (бара TensorFlow)")
    ap.add_argument("--output", default="predictions.csv", help="каде да се зачуваат предвидувањата")
    args = ap.parse_args()

    meta = load_metadata(args.models_dir)
    check_versions(meta, args.models_dir)
    feature_columns = meta.get("feature_columns") or [c for c in COL_NAMES if c not in ("label", "difficulty")]

    mode_dir = os.path.join(args.models_dir, args.mode)
    if not os.path.isdir(mode_dir):
        print("ГРЕШКА: нема папка", mode_dir, "- провери --models-dir и --mode")
        sys.exit(1)

    print("Вчитувам нов датасет:", args.data)
    X, y_raw = read_new_dataset(args.data, feature_columns)
    print("Облик на X:", X.shape, "| има labels:", y_raw is not None)

    y_true = map_labels(y_raw, args.mode) if y_raw is not None else None
    out = pd.DataFrame(index=X.index)
    if y_raw is not None:
        out["true_label_raw"] = y_raw.values
        out["true_target"] = y_true.values

    summary = []
    wanted = args.models or SKLEARN_MODELS

    # --- sklearn модели ---
    for key in wanted:
        path = os.path.join(mode_dir, key + ".joblib")
        if not os.path.exists(path):
            print("(прескокнат, нема файл:", path, ")")
            continue
        pipe = joblib.load(path)
        y_pred = pipe.predict(X)
        out["pred_" + key] = y_pred
        if y_true is not None:
            acc = evaluate(key, y_true, y_pred, mode_dir)
            summary.append({"model": key, "accuracy": round(acc, 4)})

    # --- Keras (опционо) ---
    if args.include_keras:
        kpath = os.path.join(mode_dir, "keras_nn.keras")
        ppath = os.path.join(mode_dir, "keras_preprocessor.joblib")
        if os.path.exists(kpath) and os.path.exists(ppath):
            try:
                import tensorflow as tf
                pre = joblib.load(ppath)
                net = tf.keras.models.load_model(kpath)
                Xt = pre.transform(X)
                Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
                proba = net.predict(Xt, verbose=0)
                if args.mode == "binary":
                    y_pred = (proba.ravel() >= 0.5).astype(int)
                else:
                    le = joblib.load(os.path.join(mode_dir, "keras_label_encoder.joblib"))
                    y_pred = le.inverse_transform(np.argmax(proba, axis=1))
                out["pred_keras_nn"] = y_pred
                if y_true is not None:
                    acc = evaluate("keras_nn", y_true, y_pred, mode_dir)
                    summary.append({"model": "keras_nn", "accuracy": round(acc, 4)})
            except ImportError:
                print("(--include-keras бараше TensorFlow, но не е инсталиран. pip install tensorflow)")
            except (FileNotFoundError, OSError) as e:
                print("(Keras прескокнат — недостасува/оштетен файл:", e, ")")
        else:
            print("(нема зачуван Keras модел во", mode_dir, ")")

    out.to_csv(args.output, index=False)
    print("\nПредвидувања зачувани во:", args.output)

    if summary:
        print("\n=== РЕЗИМЕ (точност на новиот датасет) ===")
        print(pd.DataFrame(summary).sort_values("accuracy", ascending=False).to_string(index=False))
    elif y_true is None:
        print("\nНовиот датасет нема labels -> само предвидувања (без евалуација).")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
