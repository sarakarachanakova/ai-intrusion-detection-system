# Систем за детекција на мрежни упади (IDS) — NSL-KDD

Машинско учење за детекција и класификација на мрежни напади (DoS, Probe, R2L, U2R) врз
датасетот **NSL-KDD**. Тренирањето се прави во **Google Colab**, моделите се **зачувуваат**, а
потоа **локално** се тестираат врз нов датасет — **без повторно тренирање**.

## Што има во папката

| Фајл | Опис |
|------|------|
| `IDS_train.ipynb` | Colab notebook: презема податоци, тренира и **зачувува** модели, прави `nsl_kdd_models.zip`. |
| `predict_local.py` | Локална скрипта: вчитува зачуван модел и тестира **нов датасет**. |
| `requirements_local.txt` | Референтни зависности за локално (точните се внатре во zip-от од Colab). |
| `_build_notebook.py` | Помошна скрипта што го генерира `IDS_train.ipynb` (не е потребна за тренирање). |

## Модели

Се тренираат во **два режима**:
- **binary** — нормален (`0`) наспроти напад (`1`)
- **multiclass** — `normal`, `DoS`, `Probe`, `R2L`, `U2R`

Алгоритми: Decision Tree, Random Forest, Logistic Regression, MLP (sklearn) + опционо длабока
невронска мрежа (Keras). Секој sklearn модел се зачувува како **цел Pipeline**
(препроцесирање + модел во едно), па локалното предвидување е директно `model.predict(нови_податоци)`.

---

## Чекор 1 — Тренирање во Google Colab

1. Отвори [Google Colab](https://colab.research.google.com/) → **File → Upload notebook** → качи `IDS_train.ipynb`.
2. **Runtime → Run all** (или ќелија по ќелија). Датасетот се презема автоматски.
3. На крај се прави `nsl_kdd_models.zip` и автоматски се нуди за **преземање**
   (ако не — преземи го рачно од панелот **Files** лево).

> Keras делот е опционен: во ќелијата под чекор 7 постави `TRAIN_KERAS = False` ако не сакаш TensorFlow.

Структура внатре во zip-от:
```
models/
├── metadata.json              # верзии, имиња на атрибути, мапирање на класи
├── requirements_local.txt     # ТОЧНИТЕ верзии од Colab (користи ги локално!)
├── summary.csv                # точност/F1 на сите модели
├── binary/
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── logistic_regression.joblib
│   ├── mlp_neural_network.joblib
│   ├── keras_nn.keras                 # ако TRAIN_KERAS=True
│   ├── keras_preprocessor.joblib
│   └── *_report.txt
└── multiclass/
    └── ... (исто + keras_label_encoder.joblib)
```

---

## Чекор 2 — Локално тестирање на нов датасет

1. Распакувај го `nsl_kdd_models.zip` во оваа папка → добиваш `models/`.
2. Направи окружување со **истите верзии** како во Colab (важно за joblib):

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r models/requirements_local.txt
```

3. Тестирај врз нов датасет (NSL-KDD формат):

```bash
# бинарна класификација
python3 predict_local.py --models-dir models --data KDDTest+.txt --mode binary

# многукласна класификација (5 категории)
python3 predict_local.py --models-dir models --data nov_dataset.csv --mode multiclass

# вклучи ја и Keras мрежата (бара tensorflow локално)
python3 predict_local.py --models-dir models --data KDDTest+.txt --mode binary --include-keras

# само одредени модели
python3 predict_local.py --models-dir models --data KDDTest+.txt --models random_forest decision_tree
```

Излез:
- табела со точност по модел (ако новиот датасет има `label` колона),
- `predictions.csv` со предвидувањата,
- `cm_<model>.png` confusion matrix за секој модел.

### Формат на новиот датасет
Истите **41 атрибут во ист редослед** како NSL-KDD. Прифатливо е:
- 43 колони (41 атрибут + `label` + `difficulty`),
- 42 колони (41 атрибут + `label`),
- 41 колони (само атрибути — тогаш само предвидување, без точност),
- CSV со заглавје што ги содржи имињата на колоните.

Непознати вредности за `service`/`flag`/`protocol_type` се толерираат
(`OneHotEncoder(handle_unknown="ignore")`).

---

## 🖥️ Демо со Gradio (за презентација)

`app_gradio.py` отвора едноставен веб-интерфејс за демонстрација пред професор — ги користи
**веќе истренираните** модели, без претренирање.

```bash
pip install -r models/requirements_local.txt   # истите верзии како Colab
pip install -r requirements_app.txt             # gradio + останати
python3 app_gradio.py
```

Потоа отвори го линкот (обично `http://127.0.0.1:7860`). Интерфејсот нуди:
- избор на **режим** (бинарна / многукласна) и **модел** (DT / RF / LR / MLP);
- **Анализа на датасет** — прикачи фајл или анализирај го вградениот `KDDTest+`, со точност,
  распределба на предвидувања и матрица на конфузија;
- **Тестирај една конекција** — зема случајна конекција од тест-сетот и дава јасна пресуда
  „⚠️ напад“ или „✅ нормално“, со споредба со вистинската ознака.

> За табот „една конекција“ во папката треба да има `KDDTest+.txt`.

---

## Често поставувано

**„Зошто да ги пинувам верзиите?“** — `joblib`/pickle моделите можат да не се вчитаат или да дадат
погрешни резултати ако локалната верзија на `scikit-learn` се разликува од таа во Colab. Затоа
notebook-от ги запишува точните верзии во `models/requirements_local.txt`.

**„Можам ли да тестирам сосема друг датасет (не NSL-KDD)?“** — Само ако има исти атрибути (исти имиња/
значење). За сосема различни податоци ќе треба повторно тренирање.

**Извор на податоци:** NSL-KDD (јавни GitHub мирори; еквивалент на Kaggle `hassan06/nslkdd`).
