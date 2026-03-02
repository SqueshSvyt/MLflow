set -e
cd "$(dirname "$0")"

echo "=== RandomForest (5 експериментів) ==="
python src/train.py --model RandomForest --max_depth 5  --n_estimators 80  --author "lab" --dataset_version "v1"
python src/train.py --model RandomForest --max_depth 10 --n_estimators 100 --author "lab" --dataset_version "v1"
python src/train.py --model RandomForest --max_depth 15 --n_estimators 120 --author "lab" --dataset_version "v1"
python src/train.py --model RandomForest --max_depth 20 --n_estimators 150 --author "lab" --dataset_version "v1"
python src/train.py --model RandomForest --max_depth 25 --n_estimators 200 --author "lab" --dataset_version "v1"

echo "=== GradientBoosting (5 експериментів) ==="
python src/train.py --model GradientBoosting --max_depth 3 --n_estimators 80  --learning_rate 0.1  --author "lab" --dataset_version "v1"
python src/train.py --model GradientBoosting --max_depth 5 --n_estimators 100 --learning_rate 0.08 --author "lab" --dataset_version "v1"
python src/train.py --model GradientBoosting --max_depth 5 --n_estimators 150 --learning_rate 0.05 --author "lab" --dataset_version "v1"
python src/train.py --model GradientBoosting --max_depth 7 --n_estimators 120 --learning_rate 0.06 --author "lab" --dataset_version "v1"
python src/train.py --model GradientBoosting --max_depth 8 --n_estimators 200 --learning_rate 0.04 --author "lab" --dataset_version "v1"

echo "=== CNN (5 експериментів) ==="
python src/train.py --model CNN --epochs 8  --batch_size 64  --learning_rate 1e-3 --author "lab" --dataset_version "v1"
python src/train.py --model CNN --epochs 10 --batch_size 64  --learning_rate 8e-4 --author "lab" --dataset_version "v1"
python src/train.py --model CNN --epochs 12 --batch_size 128 --learning_rate 1e-3 --author "lab" --dataset_version "v1"
python src/train.py --model CNN --epochs 15 --batch_size 64  --learning_rate 5e-4 --author "lab" --dataset_version "v1"
python src/train.py --model CNN --epochs 20 --batch_size 32  --learning_rate 1e-3 --author "lab" --dataset_version "v1"

echo "=== ResNet (5 експериментів) ==="
python src/train.py --model ResNet --epochs 8  --batch_size 64  --learning_rate 1e-3 --author "lab" --dataset_version "v1"
python src/train.py --model ResNet --epochs 10 --batch_size 64  --learning_rate 8e-4 --author "lab" --dataset_version "v1"
python src/train.py --model ResNet --epochs 12 --batch_size 128 --learning_rate 1e-3 --author "lab" --dataset_version "v1"
python src/train.py --model ResNet --epochs 15 --batch_size 64  --learning_rate 5e-4 --author "lab" --dataset_version "v1"
python src/train.py --model ResNet --epochs 20 --batch_size 32  --learning_rate 1e-3 --author "lab" --dataset_version "v1"

echo "Done. Start MLflow UI: mlflow ui"
echo "Filter: tags.model_type = 'RandomForest' | 'GradientBoosting' | 'CNN' | 'ResNet'"
