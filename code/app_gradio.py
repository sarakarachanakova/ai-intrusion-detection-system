#!/usr/bin/env python3
"""
Gradio кориснички интерфејс за демонстрација на IDS системот пред професор.
Ги користи ВЕЌЕ ИСТРЕНИРАНИТЕ модели од папката models/ (без претренирање).

Пуштање:
    pip install -r models/requirements_local.txt      # истите верзии како Colab (ги вчитува моделите)
    pip install -r requirements_app.txt                # gradio + matplotlib за демото
    python3 app_gradio.py
после отвори го линкот што ќе се појави (обично http://127.0.0.1:7860).
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import predict_local as pl  # повторно ги користиме готовите функции

# --------------------------------------------------------------------------
MODELS_DIR = os.environ.get("IDS_MODELS_DIR", "models")
TEST_PATH = os.environ.get("IDS_TEST_FILE", "KDDTest+.txt")

META = pl.load_metadata(MODELS_DIR)
FEATURE_COLS = META.get("feature_columns") or [c for c in pl.COL_NAMES if c not in ("label", "difficulty")]

MODEL_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "mlp_neural_network": "MLP (невронска мрежа)",
}
MODES = {"Бинарна (нормално / напад)": "binary",
         "Многукласна (DoS / Probe / R2L / U2R)": "multiclass"}

_model_cache = {}
_testset_cache = {}


def available_models(mode):
    d = os.path.join(MODELS_DIR, mode)
    if not os.path.isdir(d):
        return []
    return [k for k in pl.SKLEARN_MODELS if os.path.exists(os.path.join(d, k + ".joblib"))]


def load_model(mode, key):
    p = os.path.join(MODELS_DIR, mode, key + ".joblib")
    if p not in _model_cache:
        _model_cache[p] = joblib.load(p)
    return _model_cache[p]


def load_testset():
    if "df" not in _testset_cache:
        X, y_raw = pl.read_new_dataset(TEST_PATH, FEATURE_COLS)
        _testset_cache["df"] = (X.reset_index(drop=True),
                                None if y_raw is None else y_raw.reset_index(drop=True))
    return _testset_cache["df"]


def bin_label(v):
    return "Напад" if int(v) == 1 else "Нормално"


def pretty_pred(mode, arr):
    return [bin_label(v) for v in arr] if mode == "binary" else list(arr)


def make_cm_fig(y_true, y_pred, mode, model_key):
    labels = sorted(set(pd.unique(y_true)) | set(pd.unique(y_pred)))
    disp = [bin_label(l) for l in labels] if mode == "binary" else labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = Figure(figsize=(5.2, 4.4))     # OO API -> не се регистрира во pyplot (нема витекање)
    ax = fig.subplots()
    ConfusionMatrixDisplay(cm, display_labels=disp).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Матрица на конфузија — " + MODEL_NAMES.get(model_key, model_key))
    fig.tight_layout()
    return fig


# ------------------------- основна логика (без gradio) --------------------
def analyze_path(path, mode_label, model_key):
    """Анализира датасет на дадена патека и враќа (markdown, dataframe, figure)."""
    if not path or not os.path.exists(path):
        return "⚠️ Прикачи валиден датасет (NSL-KDD формат).", None, None
    mode = MODES.get(mode_label, "binary")
    if not model_key:
        return "⚠️ Избери модел.", None, None
    try:
        X, y_raw = pl.read_new_dataset(path, FEATURE_COLS)
    except Exception as e:
        return "❌ Грешка при читање на датасетот: %s" % e, None, None

    if len(X) == 0:
        return "⚠️ Датасетот е празен или нема валидни редови.", None, None

    try:
        pipe = load_model(mode, model_key)
        y_pred = pipe.predict(X)
    except FileNotFoundError:
        return ("⚠️ Моделот '%s' не постои за режимот '%s'. Избери друг модел."
                % (MODEL_NAMES.get(model_key, model_key), mode_label), None, None)
    except Exception as e:
        return "❌ Грешка при предвидување: %s" % e, None, None

    pred_lab = pretty_pred(mode, y_pred)
    dist = (pd.Series(pred_lab).value_counts().rename_axis("Класа")
            .reset_index(name="Број на конекции"))
    summary = ("**Модел:** %s  •  **Режим:** %s  \n**Анализирани конекции:** %d"
               % (MODEL_NAMES.get(model_key, model_key), mode_label, len(X)))
    fig = None
    if y_raw is not None:
        try:
            y_true = pl.map_labels(y_raw, mode)
            acc = accuracy_score(y_true, y_pred)
            n_attack = int(np.sum(np.asarray(y_pred) != (0 if mode == "binary" else "normal")))
            summary += ("  \n**Точност на овој датасет:** %.4f  \n"
                        "**Откриени напади (предвидени):** %d од %d" % (acc, n_attack, len(X)))
            fig = make_cm_fig(y_true, y_pred, mode, model_key)
        except Exception as e:
            summary += "  \n*(Не можат да се пресметаат метрики: %s)*" % e
    else:
        summary += "  \n*(Датасетот нема ознаки — прикажани се само предвидувања.)*"
    return summary, dist, fig


def sample_connection(mode_label, model_key):
    """Зема случајна конекција од вградениот тест-сет и ја класифицира."""
    if not os.path.exists(TEST_PATH):
        return "⚠️ Нема вграден тест-сет (%s). Стави го во истата папка." % TEST_PATH, None
    mode = MODES.get(mode_label, "binary")
    if not model_key:
        return "⚠️ Избери модел.", None
    try:
        X, y_raw = load_testset()
        i = int(np.random.randint(0, len(X)))
        row = X.iloc[[i]]
        pipe = load_model(mode, model_key)
        pred = pipe.predict(row)[0]
    except FileNotFoundError:
        return ("⚠️ Моделот '%s' не постои за режимот '%s'."
                % (MODEL_NAMES.get(model_key, model_key), mode_label), None)
    except Exception as e:
        return "❌ Грешка при предвидување: %s" % e, None

    # пресуда
    if mode == "binary":
        is_attack = int(pred) == 1
        verdict = "⚠️ НАПАД ОТКРИЕН" if is_attack else "✅ НОРМАЛЕН СООБРАЌАЈ"
    else:
        is_attack = str(pred) != "normal"
        verdict = ("⚠️ НАПАД ОТКРИЕН — тип: %s" % pred) if is_attack else "✅ НОРМАЛЕН СООБРАЌАЈ"

    md = "## %s\n**Модел:** %s\n" % (verdict, MODEL_NAMES.get(model_key, model_key))
    if y_raw is not None:
        raw = str(y_raw.iloc[i])
        fam = pl.ATTACK_MAP.get(raw, "непознато")
        true_txt = "нормално" if raw == "normal" else "%s  (фамилија: %s)" % (raw, fam)
        correct = ((not is_attack and raw == "normal") or
                   (is_attack and raw != "normal"))
        md += "**Вистинска ознака:** %s  \n**Точно предвидено:** %s\n" % (
            true_txt, "✔ да" if correct else "✘ не")

    notable = ["protocol_type", "service", "flag", "duration", "src_bytes",
               "dst_bytes", "count", "srv_count", "logged_in", "serror_rate"]
    notable = [c for c in notable if c in row.columns]
    feat = pd.DataFrame({"Атрибут": notable, "Вредност": [row.iloc[0][c] for c in notable]})
    return md, feat


# ------------------------------- gradio UI --------------------------------
def build_ui():
    import gradio as gr

    default_mode = list(MODES.keys())[0]
    model_choices = [(MODEL_NAMES[k], k) for k in available_models("binary")] or \
                    [(MODEL_NAMES[k], k) for k in pl.SKLEARN_MODELS]
    default_model = model_choices[0][1]

    with gr.Blocks(title="IDS — детекција на мрежни упади", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🛡️ Систем за детекција на мрежни упади (IDS)\n"
            "Демонстрација со **веќе истренирани** модели врз податочното множество NSL-KDD. "
            "Изберете режим и модел, па тестирајте цел датасет или една конекција.")

        with gr.Row():
            mode = gr.Radio(choices=list(MODES.keys()), value=default_mode, label="Режим на класификација")
            model = gr.Dropdown(choices=model_choices, value=default_model, label="Модел")

        with gr.Tab("📁 Анализа на датасет"):
            gr.Markdown("Прикачи датасет во NSL-KDD формат, или анализирај го вградениот **KDDTest+**.")
            file_in = gr.File(label="Датасет (.txt / .csv)", file_types=[".txt", ".csv"])
            with gr.Row():
                btn_up = gr.Button("Анализирај прикачен", variant="primary")
                btn_builtin = gr.Button("Анализирај вграден KDDTest+")
            out_summary = gr.Markdown()
            with gr.Row():
                out_dist = gr.Dataframe(label="Распределба на предвидувања", interactive=False)
                out_cm = gr.Plot(label="Матрица на конфузија")

        with gr.Tab("🔍 Тестирај една конекција"):
            gr.Markdown("Земи случајна конекција од тест-сетот и види ја пресудата на моделот.")
            btn_sample = gr.Button("🎲 Земи случаен пример од KDDTest+", variant="primary")
            out_verdict = gr.Markdown()
            out_feat = gr.Dataframe(label="Карактеристики на конекцијата", interactive=False)

        def _path(f):
            if f is None:
                return None
            return f["name"] if isinstance(f, dict) else getattr(f, "name", f)

        btn_up.click(lambda f, m, k: analyze_path(_path(f), m, k),
                     [file_in, mode, model], [out_summary, out_dist, out_cm])
        btn_builtin.click(lambda m, k: analyze_path(TEST_PATH, m, k),
                          [mode, model], [out_summary, out_dist, out_cm])
        btn_sample.click(sample_connection, [mode, model], [out_verdict, out_feat])

        def _refresh_models(mode_label):
            m = MODES.get(mode_label, "binary")
            keys = available_models(m) or list(pl.SKLEARN_MODELS)
            ch = [(MODEL_NAMES.get(k, k), k) for k in keys]
            return gr.update(choices=ch, value=(ch[0][1] if ch else None))

        mode.change(_refresh_models, [mode], [model])

    return demo


if __name__ == "__main__":
    # show_api=False го заобиколува schema-walk багот на gradio 4.44 + понов pydantic
    build_ui().launch(show_api=False)
