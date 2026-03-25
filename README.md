# MLOps — Experiment Tracking з MLflow

Проєкт з налаштування ізольованого середовища, структури ML-проєкту (Cookiecutter Data Science), Git та автоматизованого відстеження експериментів за допомогою MLflow.

## Датасет

**HAM10000** — метадані зображень діагностики шкірних утворень (Human Against Machine with 10000 training images).  
Класи: `dx` (діагноз): bkl, nv, df, mel, vasc, bcc, akiec.

У **data/raw/** зберігаються всі сирі дані:
- `HAM10000_metadata.csv` — метадані (lesion_id, image_id, dx, dx_type, age, sex, localization)
- `HAM10000_images_part_1/`, `HAM10000_images_part_2/` — зображення (.jpg)
- `hmnist_8_8_L.csv`, `hmnist_8_8_RGB.csv`, `hmnist_28_28_L.csv`, `hmnist_28_28_RGB.csv` — HMNIST (pixel-дані)

## Структура проєкту

```
mlops_lab_project/   (корінь репозиторію)
├── config/
│   ├── config.yaml      # Hydra defaults: model, hpo, paths, reproducibility
│   ├── model/
│   │   ├── random_forest.yaml
│   │   ├── gradient_boosting.yaml
│   │   ├── cnn.yaml
│   │   └── resnet.yaml
│   └── hpo/
│       ├── optuna.yaml
│       ├── random.yaml
│       └── grid.yaml
├── data/
│   ├── raw/
│   └── prepared/        # train.parquet, test.parquet, params.json
├── src/
│   ├── prepare.py
│   ├── train.py         # навчання з MLflow (argparse або Hydra)
│   ├── run_train.py     # вхід з Hydra: config + overrides
│   ├── optimize.py      # HPO (optuna | random | grid) + MLflow nested runs
│   ├── data.py
│   └── models/          # random_forest, gradient_boosting, cnn, resnet
├── mlruns/
├── dvc.yaml
└── requirements.txt
```

## Налаштування середовища

```bash
# 1. Клонування / перехід у директорію проєкту
cd MLflow

# 2. Створення віртуального середовища
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# або: venv\Scripts\activate   # Windows

# 3. Встановлення залежностей
pip install -r requirements.txt

# 4. Дані: покласти HAM10000_metadata.csv у data/raw/
# Якщо дані вже в data/ — можна скопіювати:
# cp data/HAM10000_metadata.csv data/raw/
```

## Запуск

### EDA (Jupyter)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Навчання та MLflow

```bash
# Запуск одного експерименту
python src/train.py

# Перегляд результатів у MLflow UI
mlflow ui
# Відкрити в браузері: http://localhost:5000
```

Для виконання **мінімум 5 експериментів** запустіть:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

Або вручну (усі гіперпараметри через CLI):

```bash
python src/train.py --model RandomForest --max_depth 5 --n_estimators 100 --author "student" --dataset_version "v1"
python src/train.py --model LogisticRegression --C 1.0
python src/train.py --model GradientBoosting --max_depth 5 --n_estimators 100 --learning_rate 0.1
```

**Моделі:** `RandomForest`, `GradientBoosting`, `CNN`, `ResNet` (аргумент `--model`).

**CLI-аргументи:** `--model`, `--max_depth`, `--n_estimators`, `--learning_rate` (GB/CNN/ResNet), `--epochs`, `--batch_size`, `--test_size`, `--random_state`, `--author`, `--dataset_version`.

### Hydra: параметризація пайплайна

```bash
# Навчання з конфігу (defaults: model=random_forest)
python src/run_train.py

# Інша модель або перевизначення параметрів
python src/run_train.py model=gradient_boosting
python src/run_train.py model.max_depth=8 model.n_estimators=150
python src/run_train.py reproducibility.random_state=123
```

### HPO: optimize.py (optuna | random | grid) + MLflow nested runs

Підбір гіперпараметрів: **parent run** = study, **child runs** = trials. Метрика на **validation**; фінальна модель тренується на повному train, оцінка на test.

```bash
# За замовчуванням: model=random_forest, hpo=optuna
python src/optimize.py

# Random Search або Grid Search
python src/optimize.py hpo=random
python src/optimize.py hpo=grid

# Інша модель
python src/optimize.py model=gradient_boosting hpo=optuna
python src/optimize.py model=cnn hpo.n_trials=8   # CNN: Optuna, менше trials (довше на trial)
```

У MLflow: parent run з тегом `run_type=hpo_study`, `hpo_strategy=optuna|random|grid`; етап `optimize` у `dvc.yaml` запускає `python src/optimize.py`.

### Крок 7. Реєстрація найкращої моделі (додатковий)

Очікуваний результат:

- **(B) Завжди:** найкраща модель записана в MLflow як артефакт `best_model` у parent run. Відтворення: `mlflow.sklearn.load_model(f"runs:/<run_id>/best_model")` або через UI → Run → Artifacts → best_model.
- **(A) За наявності MLflow tracking server з backend store:** увімкніть реєстрацію в конфігу та переведіть версію в Staging:
  ```yaml
  # config/config.yaml або override
  model_registry:
    register_best: true
    registered_model_name: ham10000_best
    stage: Staging
  ```
  Або з CLI: `python src/optimize.py model_registry.register_best=true`. Після запуску в UI з’явиться Models → ham10000_best → версія в Staging.

### Відтворюваність (академічна доброчесність)

У кожному run логуються: `random_state`, `dataset_version`, `test_size`, `split_random_state`; якщо проєкт у Git — тег `code_commit` (hash). Конфіг запуску можна зберегти як артефакт. Не змінюйте метрики вручну; у висновках пояснюйте отримані значення.

### Аналіз результатів та запуск HPO

Підсумок runs дивіться у **MLflow UI** (`mlflow ui` → експеримент `ham10000_baseline`, Compare / фільтри).

```bash
# Оптимізація гіперпараметрів: RandomForest (за замовчуванням, 20 trials)
python src/optimize.py

# Оптимізація для CNN (Optuna, менше trials — кожен trial довший)
python src/optimize.py model=cnn hpo.n_trials=8
```

### Перевірка результатів (MLflow UI)

1. Запустіть інтерфейс: `mlflow ui`
2. Відкрийте браузер: **http://127.0.0.1:5000**
3. Переконайтеся:
   - Є експеримент **ham10000_baseline** і список Runs.
   - У кожному run: параметри, метрики **train** та **test** (accuracy, f1_weighted), артефакт **feature_importance.png**, теги.
4. **Compare:** виділіть усі запуски → кнопка **Compare** → побудуйте графік: по осі Y метрика (наприклад `test_accuracy` або `test_f1_weighted`), по осі X параметр `params.max_depth`. Аналіз кривих допомагає побачити overfitting (train >> test при великому max_depth).
5. **Search (фільтр за тегами):** у полі пошуку введіть, наприклад:
   - `tags.model_type = "RandomForest"`
   - `tags.author = "lab"`
   щоб відфільтрувати запуски за тегами.

### CI / GitHub Actions (CML) і тести

- **Workflow:** `.github/workflows/cml.yaml` — на `push` / `pull_request` (гілки `main` / `master`): встановлення залежностей (`requirements-ci.txt`), **black** / **flake8** на `tests/` та `src/ci_export_rf.py`, **pytest** (швидкі тести), синтетичні дані (`tests/fixtures/generate_mini_ham10000.py`), **prepare → train** (RandomForest), інтеграційні тести (**Quality Gate** за `test_f1_weighted`, змінна `QUALITY_F1_MIN`), експорт **`ci_artifacts/`** (`model.pkl`, `metrics.json`, `confusion_matrix.png`), **CML**-звіт у PR (`cml comment create`). На **main/master** після успіху завантажується артефакт **`model-ci-bundle`**.
- **Локально:** `pip install -r requirements.txt -r requirements-dev.txt` (або `requirements-ci.txt` для мінімального набору), `pytest tests/ -m "not integration"`, після повного пайплайну — `pytest tests/ -m integration`.
- **MLflow Registry (опційно CD):** залиште реєстрацію через `config/model_registry` у `optimize.py` або додайте крок у workflow з секретами сервера MLflow.
