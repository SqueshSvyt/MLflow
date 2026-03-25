"""
Models package: one subpackage per model type.
Each model has its own folder with train.py that implements run().
"""
from . import random_forest
from . import gradient_boosting
from . import cnn
from . import resnet

MODEL_REGISTRY = {
    "RandomForest": random_forest,
    "GradientBoosting": gradient_boosting,
    "CNN": cnn,
    "ResNet": resnet,
}

__all__ = ["MODEL_REGISTRY", "random_forest", "gradient_boosting", "cnn", "resnet"]
