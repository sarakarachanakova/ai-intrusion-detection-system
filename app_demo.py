#!/usr/bin/env python3
"""
Комбинирано Gradio демо за IDS-системот — работи и со NSL-KDD и со UNSW-NB15,
со истите веќе истренирани модели (без претренирање).

Три начини да се тестира еден пример:
  1. Анализа на цел датасет (прикачен фајл или вграден тест-сет)
  2. Случајна конекција земена од тест-сетот
  3. Рачно внесени сопствени вредности (уредлива табела, преполнета со типични
     вредности што може да се менуваат пред предвидување)

Пуштање (од оваа папка, dtvi-project/):
    "<патека до venv>/bin/python3" app_demo.py
после отвори http://127.0.0.1:7860
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "code"))
import predict_local as pl  # NSL-KDD хелпери (COL_NAMES, ATTACK_MAP, read_new_dataset...)

MODEL_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "mlp_neural_network": "MLP (невронска мрежа)",
}
MODES = {"Бинарна (нормално / напад)": "binary",
         "Многукласна (по тип на напад)": "multiclass"}
SKLEARN_MODELS = list(MODEL_NAMES.keys())

UNSW_CAT_COLS = ["proto", "service", "state"]

_cache = {}


# ===========================================================================
# NSL-KDD ============================================================
# ===========================================================================
def nsl_load_test():
    key = "nsl_test"
    if key not in _cache:
        path = os.path.join(ROOT, "code", "KDDTest+.txt")
        X, y_raw = pl.read_new_dataset(path, pl.COL_NAMES[:-2])
        y_bin = (y_raw.astype(str) != "normal").astype(int)
        y_multi = pl.map_labels(y_raw, "multiclass")
        _cache[key] = (X.reset_index(drop=True), y_raw.reset_index(drop=True),
                       y_bin.reset_index(drop=True), y_multi.reset_index(drop=True))
    return _cache[key]


def nsl_defaults():
    key = "nsl_defaults"
    if key not in _cache:
        X, y_raw, *_ = nsl_load_test()
        Xn = X[y_raw.astype(str) == "normal"]  # типична НОРМАЛНА конекција, не просек од сѐ
        row = {}
        for c in X.columns:
            if c in ("protocol_type", "service", "flag"):
                row[c] = Xn[c].mode().iloc[0]
            else:
                row[c] = float(pd.to_numeric(Xn[c], errors="coerce").median())
        _cache[key] = row
    return dict(_cache[key])


# ===========================================================================
# UNSW-NB15 ===========================================================
# ===========================================================================
def unsw_load_test():
    key = "unsw_test"
    if key not in _cache:
        path = os.path.join(ROOT, "code_unsw", "data", "UNSW_NB15_testing-set.parquet")
        df = pd.read_parquet(path)
        X = df.drop(columns=["attack_cat", "label"]).reset_index(drop=True)
        y_bin = df["label"].astype(int).reset_index(drop=True)
        y_multi = df["attack_cat"].astype(str).reset_index(drop=True)
        _cache[key] = (X, y_multi.copy(), y_bin, y_multi)  # y_raw slot = attack_cat (за приказ)
    return _cache[key]


def unsw_defaults():
    key = "unsw_defaults"
    if key not in _cache:
        path = os.path.join(ROOT, "code_unsw", "data", "UNSW_NB15_training-set.parquet")
        df = pd.read_parquet(path)
        Xn = df[df["label"] == 0].drop(columns=["attack_cat", "label"])  # типична НОРМАЛНА конекција
        row = {}
        for c in Xn.columns:
            if c in UNSW_CAT_COLS:
                row[c] = Xn[c].mode().iloc[0]
            else:
                row[c] = float(pd.to_numeric(Xn[c], errors="coerce").median())
        _cache[key] = row
    return dict(_cache[key])


# ===========================================================================
# Општ регистар на датасети  ================================================
# ===========================================================================
DATASETS = {
    "NSL-KDD": {
        "models_dir": os.path.join(ROOT, "code", "models"),
        "load_test": nsl_load_test,
        "defaults": nsl_defaults,
        "cat_cols": ["protocol_type", "service", "flag"],
        "bin_labels": {0: "нормално", 1: "напад"},
        "true_label_fmt": lambda raw: ("нормално" if str(raw) == "normal"
                                        else "%s  (фамилија: %s)" % (raw, pl.ATTACK_MAP.get(str(raw), "непознато"))),
    },
    "UNSW-NB15": {
        "models_dir": os.path.join(ROOT, "code_unsw", "models"),
        "load_test": unsw_load_test,
        "defaults": unsw_defaults,
        "cat_cols": UNSW_CAT_COLS,
        "bin_labels": {0: "нормално", 1: "напад"},
        "true_label_fmt": lambda raw: ("нормално" if str(raw) == "Normal" else str(raw)),
    },
}


def available_models(dataset, mode):
    d = os.path.join(DATASETS[dataset]["models_dir"], mode)
    if not os.path.isdir(d):
        return []
    return [k for k in SKLEARN_MODELS if os.path.exists(os.path.join(d, k + ".joblib"))]


def load_model(dataset, mode, key):
    p = os.path.join(DATASETS[dataset]["models_dir"], mode, key + ".joblib")
    ck = ("model", p)
    if ck not in _cache:
        _cache[ck] = joblib.load(p)
    return _cache[ck]


def bin_label(dataset, v):
    return DATASETS[dataset]["bin_labels"].get(int(v), str(v))


def pretty_pred(dataset, mode, arr):
    return [bin_label(dataset, v) for v in arr] if mode == "binary" else [str(v) for v in arr]


def make_cm_fig(dataset, y_true, y_pred, mode, model_key):
    labels = sorted(set(pd.unique(y_true)) | set(pd.unique(y_pred)), key=str)
    disp = [bin_label(dataset, l) for l in labels] if mode == "binary" else [str(l) for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = Figure(figsize=(5.4, 4.6))
    ax = fig.subplots()
    ConfusionMatrixDisplay(cm, display_labels=disp).plot(ax=ax, cmap="Blues", colorbar=False,
                                                            xticks_rotation=45 if mode != "binary" else "horizontal")
    ax.set_title("Матрица на конфузија — " + MODEL_NAMES.get(model_key, model_key))
    fig.tight_layout()
    return fig


# --------------------------- Таб 1: анализа на датасет ---------------------
def analyze_dataset(dataset, mode_label, model_key):
    mode = MODES.get(mode_label, "binary")
    if not model_key:
        return "⚠️ Избери модел.", None, None
    try:
        X, y_raw, y_bin, y_multi = DATASETS[dataset]["load_test"]()
        y_true = y_bin if mode == "binary" else y_multi
        pipe = load_model(dataset, mode, model_key)
        y_pred = pipe.predict(X)
    except FileNotFoundError:
        return "⚠️ Моделот не постои за овој режим/датасет.", None, None
    except Exception as e:
        return "❌ Грешка: %s" % e, None, None

    pred_lab = pretty_pred(dataset, mode, y_pred)
    dist = (pd.Series(pred_lab).value_counts().rename_axis("Класа")
            .reset_index(name="Број на конекции"))
    acc = accuracy_score(y_true, y_pred)
    if mode == "binary":
        n_attack = int(np.sum(np.asarray(y_pred) != 0))
    else:
        n_attack = int(np.sum([str(p).lower() != "normal" for p in y_pred]))
    summary = ("**Датасет:** %s  •  **Модел:** %s  •  **Режим:** %s  \n"
               "**Тестирани конекции:** %d  \n**Точност:** %.4f  \n**Откриени напади:** %d од %d"
               % (dataset, MODEL_NAMES.get(model_key, model_key), mode_label, len(X), acc, n_attack, len(X)))
    fig = make_cm_fig(dataset, y_true, y_pred, mode, model_key)
    return summary, dist, fig


# --------------------- Таб 2: случајна конекција од тестот -----------------
def sample_connection(dataset, mode_label, model_key):
    mode = MODES.get(mode_label, "binary")
    if not model_key:
        return "⚠️ Избери модел.", None
    try:
        X, y_raw, y_bin, y_multi = DATASETS[dataset]["load_test"]()
        i = int(np.random.randint(0, len(X)))
        row = X.iloc[[i]]
        pipe = load_model(dataset, mode, model_key)
        pred = pipe.predict(row)[0]
    except FileNotFoundError:
        return "⚠️ Моделот не постои за овој режим/датасет.", None
    except Exception as e:
        return "❌ Грешка: %s" % e, None

    if mode == "binary":
        is_attack = int(pred) == 1
        verdict = "⚠️ НАПАД ОТКРИЕН" if is_attack else "✅ НОРМАЛЕН СООБРАЌАЈ"
    else:
        is_attack = str(pred).lower() != "normal"
        verdict = ("⚠️ НАПАД ОТКРИЕН — тип: %s" % pred) if is_attack else "✅ НОРМАЛЕН СООБРАЌАЈ"

    raw = y_raw.iloc[i]
    true_txt = DATASETS[dataset]["true_label_fmt"](raw)
    true_is_attack = str(raw).lower() not in ("normal",)
    correct = (is_attack == true_is_attack)
    md = ("## %s\n**Датасет:** %s  •  **Модел:** %s\n\n"
          "**Вистинска ознака:** %s  \n**Точно предвидено:** %s\n"
          % (verdict, dataset, MODEL_NAMES.get(model_key, model_key), true_txt, "✔ да" if correct else "✘ не"))

    cols = list(row.columns)[:12]
    feat = pd.DataFrame({"Атрибут": cols, "Вредност": [row.iloc[0][c] for c in cols]})
    return md, feat


# --------------------- Таб 3: рачно внесени вредности -----------------------
def default_row_df(dataset):
    d = DATASETS[dataset]["defaults"]()
    return pd.DataFrame([d])


def manual_predict(dataset, mode_label, model_key, edited_df):
    mode = MODES.get(mode_label, "binary")
    if not model_key:
        return None
    if edited_df is None or len(edited_df) == 0:
        return None
    cat_cols = DATASETS[dataset]["cat_cols"]
    X = edited_df.copy()
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        else:
            X[c] = X[c].astype(str)
    try:
        pipe = load_model(dataset, mode, model_key)
        pred = pipe.predict(X)
    except FileNotFoundError:
        out = edited_df.copy()
        out["Предвидување"] = "⚠️ моделот не постои за овој режим"
        return out
    except Exception as e:
        out = edited_df.copy()
        out["Предвидување"] = "❌ грешка: %s" % e
        return out

    out = edited_df.copy()
    if mode == "binary":
        out["Предвидување"] = [("⚠️ напад" if int(p) == 1 else "✅ нормално") for p in pred]
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X)
                classes = list(pipe.classes_)
                idx = classes.index(1)
                out["Веројатност за напад"] = [round(float(p[idx]), 3) for p in proba]
            except Exception:
                pass
    else:
        out["Предвидување"] = [("✅ нормално" if str(p).lower() == "normal" else "⚠️ напад — %s" % p) for p in pred]
    return out


# ================================= UI =======================================
def build_ui():
    import gradio as gr

    default_dataset = "NSL-KDD"
    default_mode = list(MODES.keys())[0]
    model_choices = [(MODEL_NAMES[k], k) for k in available_models(default_dataset, "binary")] or \
                    [(MODEL_NAMES[k], k) for k in SKLEARN_MODELS]
    default_model = model_choices[0][1]

    with gr.Blocks(title="IDS — детекција на мрежни упади", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🛡️ Систем за детекција на мрежни упади (IDS)\n"
            "Демонстрација со **веќе истренирани** модели, врз **NSL-KDD** и **UNSW-NB15**. "
            "Избери датасет, режим и модел, па тестирај цел датасет, случајна конекција, "
            "или внеси сопствени вредности.")

        with gr.Row():
            dataset = gr.Radio(choices=list(DATASETS.keys()), value=default_dataset, label="Множество податоци")
            mode = gr.Radio(choices=list(MODES.keys()), value=default_mode, label="Режим на класификација")
            model = gr.Dropdown(choices=model_choices, value=default_model, label="Модел")

        with gr.Tab("📁 Анализа на датасет"):
            gr.Markdown("Анализирај го вградениот тест-сет за избраното множество податоци.")
            btn_builtin = gr.Button("Анализирај вграден тест-сет", variant="primary")
            out_summary = gr.Markdown()
            with gr.Row():
                out_dist = gr.Dataframe(label="Распределба на предвидувања", interactive=False)
                out_cm = gr.Plot(label="Матрица на конфузија")

        with gr.Tab("🎲 Случајна конекција"):
            gr.Markdown("Земи случајна конекција од тест-сетот и види ја пресудата на моделот.")
            btn_sample = gr.Button("🎲 Земи случаен пример", variant="primary")
            out_verdict = gr.Markdown()
            out_feat = gr.Dataframe(label="Карактеристики на конекцијата (првите 12)", interactive=False)

        with gr.Tab("✏️ Внеси сопствени вредности"):
            gr.Markdown(
                "Табелата е преполнета со типични (медијана/најчести) вредности за избраното "
                "множество податоци. Измени ги ќелиите по желба (може и повеќе редови) и притисни "
                "„Предвиди“. Категоријалните колони (протокол/сервис/статус) прифаќаат текст.")
            btn_reset = gr.Button("↺ Вчитај урнек за избраното множество")
            manual_in = gr.Dataframe(value=default_row_df(default_dataset), interactive=True, wrap=False,
                                      label="Твои вредности (уреди слободно)")
            btn_predict = gr.Button("▶ Предвиди", variant="primary")
            manual_out = gr.Dataframe(label="Резултат", interactive=False, wrap=False)

        # ---- callbacks ----
        def _refresh_models(dataset_v, mode_label):
            m = MODES.get(mode_label, "binary")
            keys = available_models(dataset_v, m) or list(SKLEARN_MODELS)
            ch = [(MODEL_NAMES.get(k, k), k) for k in keys]
            return gr.update(choices=ch, value=(ch[0][1] if ch else None))

        dataset.change(_refresh_models, [dataset, mode], [model])
        mode.change(_refresh_models, [dataset, mode], [model])
        dataset.change(lambda d: default_row_df(d), [dataset], [manual_in])

        btn_builtin.click(analyze_dataset, [dataset, mode, model], [out_summary, out_dist, out_cm])
        btn_sample.click(sample_connection, [dataset, mode, model], [out_verdict, out_feat])
        btn_reset.click(lambda d: default_row_df(d), [dataset], [manual_in])
        btn_predict.click(manual_predict, [dataset, mode, model, manual_in], [manual_out])

    return demo


if __name__ == "__main__":
    build_ui().launch(show_api=False)
