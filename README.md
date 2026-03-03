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
mlops_lab_1/   (або корінь репозиторію)
├── .gitignore
├── requirements.txt
├── README.md
├── venv/                 # віртуальне середовище (не в Git)
├── data/
│   └── raw/              # сирі дані (не в Git)
│       ├── HAM10000_metadata.csv
│       ├── HAM10000_images_part_1/, HAM10000_images_part_2/
│       └── hmnist_*.csv
├── notebooks/
│   └── 01_eda.ipynb      # EDA
├── src/
│   └── train.py          # скрипт навчання з MLflow
├── mlruns/               # логи MLflow (не в Git)
└── models/               # збережені моделі (не в Git)
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

**Моделі:** `RandomForest`, `LogisticRegression`, `GradientBoosting` (аргумент `--model`).

**CLI-аргументи:** `--model`, `--max_depth`, `--n_estimators`, `--learning_rate` (GB), `--C`, `--max_iter` (LR), `--test_size`, `--random_state`, `--author`, `--dataset_version`.

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
