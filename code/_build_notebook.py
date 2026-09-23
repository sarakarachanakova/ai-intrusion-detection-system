"""Builds IDS_train.ipynb (Google Colab notebook) from the cell definitions below.
Run:  python3 _build_notebook.py
This avoids the nbformat dependency; the notebook JSON is assembled by hand.
"""
import json

cells = []

def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })

def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })

# ---------------------------------------------------------------------------
md("""# Систем за детекција на мрежни упади (IDS) — NSL-KDD
**Студент:** Сара Карачанакова · **Предмет:** Дигитална трансформација и вештачка интелигенција

Овој notebook:
1. Го презема јавниот **NSL-KDD** датасет (без потреба од Kaggle најава).
2. Тренира 4 класични модели (Decision Tree, Random Forest, Logistic Regression, MLP) + опционо длабока невронска мрежа (Keras).
3. Тренира **бинарна** (нормален vs напад) **и многукласна** (normal / DoS / Probe / R2L / U2R) класификација.
4. Ги **зачувува сите модели заедно со препроцесирањето** во `models/` и прави `nsl_kdd_models.zip` за преземање.
5. Зачуваниот zip потоа го распакуваш локално и со `predict_local.py` тестираш **нов датасет на веќе истрениран модел** — без повторно тренирање.

> Сите модели се зачувани како **sklearn Pipeline** (препроцесирање + модел во едно), па локалното вчитување е директно `joblib.load(...)` и `model.predict(new_data)` — без рачно енкодирање/скалирање.""")

# ---------------------------------------------------------------------------
md("""## 1. Импорти и верзии
Верзиите се испишуваат и се зачувуваат во `metadata.json` — **локално мора да ги користиш истите верзии** за да се вчитаат моделите коректно (joblib pickle е чувствителен на верзија на scikit-learn).""")

code("""import sys, sklearn, numpy as np, pandas as pd, scipy, joblib
print("Python      :", sys.version.split()[0])
print("scikit-learn:", sklearn.__version__)
print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("scipy       :", scipy.__version__)
print("joblib      :", joblib.__version__)""")

# ---------------------------------------------------------------------------
md("""## 2. Преземање на NSL-KDD
Извор: јавни GitHub мирори на NSL-KDD (еквивалент на Kaggle `hassan06/nslkdd`).
- `KDDTrain+.txt` — тренинг множество
- `KDDTest+.txt`  — тест множество (содржи и типови напади што ги нема во тренинг — реален тест на генерализација)""")

code("""import os, urllib.request, ssl

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Public mirrors (no Kaggle auth needed). %2B = '+' encoded for the URL.
MIRRORS = {
    "KDDTrain+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTrain%2B.txt",
    ],
    "KDDTest+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTest%2B.txt",
    ],
}

def download(fname, urls):
    dest = os.path.join(DATA_DIR, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("Already downloaded:", dest)
        return dest
    last_err = None
    for url in urls:
        try:
            print("Downloading", url, "...")
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(url, context=ctx, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            if os.path.getsize(dest) > 0:
                print("  saved ->", dest, os.path.getsize(dest), "bytes")
                return dest
        except Exception as e:
            last_err = e
            print("  failed:", e)
    raise RuntimeError("Could not download %s: %s" % (fname, last_err))

for fname, urls in MIRRORS.items():
    download(fname, urls)""")

# ---------------------------------------------------------------------------
md("""## 3. Вчитување на податоците + дефиниции
NSL-KDD има 41 атрибут + `label` (тип на сообраќај) + `difficulty` (го отфрламе).
`ATTACK_MAP` ги групира поединечните напади во 5 категории (normal, DoS, Probe, R2L, U2R).""")

code('''COL_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty",
]

CAT_COLS = ["protocol_type", "service", "flag"]

# Map each raw attack name -> one of 5 classes. Covers every label in KDDTrain+ and KDDTest+.
ATTACK_MAP = {
    "normal": "normal",
    # DoS
    "back":"DoS","land":"DoS","neptune":"DoS","pod":"DoS","smurf":"DoS","teardrop":"DoS",
    "apache2":"DoS","udpstorm":"DoS","processtable":"DoS","worm":"DoS","mailbomb":"DoS",
    # Probe
    "satan":"Probe","ipsweep":"Probe","nmap":"Probe","portsweep":"Probe","mscan":"Probe","saint":"Probe",
    # R2L
    "guess_passwd":"R2L","ftp_write":"R2L","imap":"R2L","phf":"R2L","multihop":"R2L",
    "warezmaster":"R2L","warezclient":"R2L","spy":"R2L","xlock":"R2L","xsnoop":"R2L",
    "snmpguess":"R2L","snmpgetattack":"R2L","httptunnel":"R2L","sendmail":"R2L","named":"R2L",
    # U2R
    "buffer_overflow":"U2R","loadmodule":"U2R","rootkit":"U2R","perl":"U2R","sqlattack":"U2R",
    "xterm":"U2R","ps":"U2R",
}

def load_nslkdd(path):
    df = pd.read_csv(path, names=COL_NAMES)
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])
    return df

df_train = load_nslkdd("data/KDDTrain+.txt")
df_test  = load_nslkdd("data/KDDTest+.txt")

FEATURE_COLS = [c for c in df_train.columns if c != "label"]   # 41 features
NUM_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

print("Train:", df_train.shape, " Test:", df_test.shape)
print("Features:", len(FEATURE_COLS), "| categorical:", CAT_COLS)
df_train.head()''')

# ---------------------------------------------------------------------------
md("""## 4. Брза анализа на податоците (EDA)""")

code('''import matplotlib.pyplot as plt

print("Тренинг — дистрибуција по категорија:")
print(df_train["label"].map(lambda x: ATTACK_MAP.get(x, "unknown")).value_counts())

top = df_train["label"].value_counts().head(15)
plt.figure(figsize=(12, 4))
top.plot(kind="bar", title="Топ 15 типови сообраќај/напади (тренинг)")
plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------------------
md("""## 5. Препроцесирање и дефиниција на моделите
`StandardScaler` за нумеричките + `OneHotEncoder(handle_unknown="ignore")` за категоричките.
`handle_unknown="ignore"` е клучно: ако новиот датасет содржи непознат `service`/`flag`, моделот нема да падне.""")

code('''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])

def build_models():
    return {
        "decision_tree":        DecisionTreeClassifier(random_state=42),
        "random_forest":        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "logistic_regression":  LogisticRegression(max_iter=1000),
        "mlp_neural_network":   MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
    }

def make_xy(df, mode):
    """Build X (raw features) and y (target) for a given mode."""
    X = df[FEATURE_COLS].copy()
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    if mode == "binary":
        y = (df["label"] != "normal").astype(int)          # 0 = normal, 1 = attack
    else:
        y = df["label"].map(lambda x: ATTACK_MAP.get(x, "unknown"))
    return X, y''')

# ---------------------------------------------------------------------------
md("""## 6. Тренирање, евалуација и **зачувување** на моделите
Тренираме за двата режима (`binary` и `multiclass`). Секој модел се зачувува како комплетен Pipeline во `models/<mode>/<model>.joblib`.
Дополнително се зачувуваат `metadata.json`, `requirements_local.txt` (точни верзии) и текстуални извештаи.""")

code('''import json, time
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODELS_ROOT = "models"
MODES = ["binary", "multiclass"]
summary_rows = []
trained = {}   # (mode, key) -> fitted pipeline, kept for the plots below

for mode in MODES:
    out_dir = os.path.join(MODELS_ROOT, mode)
    os.makedirs(out_dir, exist_ok=True)
    X_tr, y_tr = make_xy(df_train, mode)
    X_te, y_te = make_xy(df_test, mode)
    print("\\n" + "="*70 + "\\nMODE:", mode, "\\n" + "="*70)

    for key, clf in build_models().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("clf", clf)])
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        secs = time.time() - t0
        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        f1 = (f1_score(y_te, y_pred, pos_label=1, zero_division=0) if mode == "binary"
              else f1_score(y_te, y_pred, average="macro", zero_division=0))

        path = os.path.join(out_dir, key + ".joblib")
        joblib.dump(pipe, path)
        trained[(mode, key)] = pipe

        rep = classification_report(y_te, y_pred, digits=4, zero_division=0)
        with open(os.path.join(out_dir, key + "_report.txt"), "w") as f:
            f.write("MODE=%s  MODEL=%s\\nAccuracy=%.4f  F1=%.4f  train_time=%.1fs\\n\\n%s"
                    % (mode, key, acc, f1, secs, rep))

        summary_rows.append({"mode": mode, "model": key,
                             "accuracy": round(acc, 4), "f1": round(f1, 4),
                             "train_s": round(secs, 1)})
        print("  %-22s acc=%.4f  f1=%.4f  (%.1fs)  -> %s" % (key, acc, f1, secs, path))

summary_df = pd.DataFrame(summary_rows)
print("\\n=== РЕЗИМЕ ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(MODELS_ROOT, "summary.csv"), index=False)''')

# ---------------------------------------------------------------------------
md("""### 6.1 Зачувување на `metadata.json` и `requirements_local.txt`
`requirements_local.txt` ги содржи **точните верзии од Colab** — локално инсталирај со `pip install -r requirements_local.txt` за да нема несовпаѓање.""")

code('''metadata = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "dataset": "NSL-KDD (KDDTrain+ / KDDTest+)",
    "modes": MODES,
    "models": list(build_models().keys()),
    "label_column": "label",
    "feature_columns": FEATURE_COLS,
    "categorical_columns": CAT_COLS,
    "numeric_columns": NUM_COLS,
    "binary_mapping": {"0": "normal", "1": "attack"},
    "attack_map": ATTACK_MAP,
    "versions": {
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    },
}
with open(os.path.join(MODELS_ROOT, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

reqs = [
    "scikit-learn==" + sklearn.__version__,
    "numpy==" + np.__version__,
    "pandas==" + pd.__version__,
    "scipy==" + scipy.__version__,
    "joblib==" + joblib.__version__,
]
with open(os.path.join(MODELS_ROOT, "requirements_local.txt"), "w") as f:
    f.write("\\n".join(reqs) + "\\n")

print("Saved metadata.json and requirements_local.txt")
print("\\n".join(reqs))''')

# ---------------------------------------------------------------------------
md("""## 7. (Опционо) Длабока невронска мрежа со Keras
Бидејќи Keras не влегува во sklearn Pipeline, го зачувуваме моделот (`.keras`) + посебно фитнатиот препроцесор (`keras_preprocessor.joblib`).
Постави `TRAIN_KERAS = False` ако сакаш да го прескокнеш (тогаш не ти треба TensorFlow локално).""")

code('''TRAIN_KERAS = True

if TRAIN_KERAS:
    import tensorflow as tf
    from tensorflow.keras import layers, models as kmodels
    from sklearn.preprocessing import LabelEncoder
    print("TensorFlow:", tf.__version__)

    for mode in MODES:
        out_dir = os.path.join(MODELS_ROOT, mode)
        X_tr, y_tr = make_xy(df_train, mode)
        X_te, y_te = make_xy(df_test, mode)

        pre = build_preprocessor()
        Xtr = pre.fit_transform(X_tr)
        Xte = pre.transform(X_te)
        Xtr = Xtr.toarray() if hasattr(Xtr, "toarray") else np.asarray(Xtr)
        Xte = Xte.toarray() if hasattr(Xte, "toarray") else np.asarray(Xte)
        joblib.dump(pre, os.path.join(out_dir, "keras_preprocessor.joblib"))

        if mode == "binary":
            net = kmodels.Sequential([
                layers.Input(shape=(Xtr.shape[1],)),
                layers.Dense(128, activation="relu"), layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ])
            net.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            net.fit(Xtr, y_tr.values, epochs=20, batch_size=256, validation_split=0.2, verbose=2)
            loss, acc = net.evaluate(Xte, y_te.values, verbose=0)
        else:
            le = LabelEncoder().fit(y_tr)
            joblib.dump(le, os.path.join(out_dir, "keras_label_encoder.joblib"))
            ytr = le.transform(y_tr)
            net = kmodels.Sequential([
                layers.Input(shape=(Xtr.shape[1],)),
                layers.Dense(128, activation="relu"), layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dense(len(le.classes_), activation="softmax"),
            ])
            net.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            net.fit(Xtr, ytr, epochs=20, batch_size=256, validation_split=0.2, verbose=2)
            mask = pd.Series(y_te).isin(le.classes_).values
            yte = le.transform(pd.Series(y_te)[mask])
            loss, acc = net.evaluate(Xte[mask], yte, verbose=0)

        net.save(os.path.join(out_dir, "keras_nn.keras"))
        print("  [%s] Keras NN test accuracy=%.4f  -> %s" % (mode, acc, out_dir))
else:
    print("Keras прескокнато (TRAIN_KERAS=False).")''')

# ---------------------------------------------------------------------------
md("""## 8. Визуелизации (confusion matrix, ROC, важност на атрибути)""")

code('''import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

# --- 8.1 Confusion matrix: multiclass Random Forest ---
mode = "multiclass"
pipe = trained[(mode, "random_forest")]
X_te, y_te = make_xy(df_test, mode)
y_pred = pipe.predict(X_te)
labels = sorted(set(pd.unique(y_te)) | set(pd.unique(y_pred)))
cm = confusion_matrix(y_te, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — Random Forest (multiclass)")
plt.tight_layout(); plt.show()

# --- 8.2 ROC за бинарните модели ---
mode = "binary"
X_te, y_te = make_xy(df_test, mode)
plt.figure(figsize=(8, 6))
for key in build_models():
    p = trained[(mode, key)]
    if hasattr(p, "predict_proba"):
        prob = p.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, prob)
        plt.plot(fpr, tpr, label="%s (AUC=%.3f)" % (key, auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], "k--", label="Random (0.5)")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC — бинарна класификација"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()

# --- 8.3 Топ 20 најважни атрибути (RF, бинарно) ---
pipe = trained[("binary", "random_forest")]
ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"]
feat_names = NUM_COLS + list(ohe.get_feature_names_out(CAT_COLS))
imp = pipe.named_steps["clf"].feature_importances_
order = np.argsort(imp)[::-1][:20]
plt.figure(figsize=(9, 6))
plt.barh(np.array(feat_names)[order][::-1], imp[order][::-1])
plt.title("Топ 20 атрибути (Random Forest, бинарно)")
plt.tight_layout(); plt.show()''')

# ---------------------------------------------------------------------------
md("""## 9. Спакување и преземање на моделите
Се прави `nsl_kdd_models.zip` со сè во `models/`. Преземи го, распакувај го локално и користи `predict_local.py`.""")

code('''import shutil
archive = shutil.make_archive("nsl_kdd_models", "zip", "models")
print("Created:", archive, os.path.getsize(archive), "bytes")

# Listing на содржината
for root, _, fs in os.walk("models"):
    for fn in sorted(fs):
        print(" ", os.path.join(root, fn))

try:
    from google.colab import files
    files.download(archive)
except Exception as e:
    print("\\n(Не сум во Colab или преземањето не е достапно:", e, ")")
    print("Преземи го 'nsl_kdd_models.zip' рачно од панелот Files лево.")''')

# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("IDS_train.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Wrote IDS_train.ipynb with", len(cells), "cells")
