"""Smoke test loading the committed VGG model with the real Keras/JAX stack.

Proves only that deserialization and a single CPU inference work. Does not
establish diagnostic accuracy or medical validity.
"""
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from keras.models import load_model

ROOT = Path(__file__).resolve().parents[1]


def test_vgg_loads_and_predicts_a_four_class_vector():
    path = ROOT / "model_1" / "Vgg.h5"
    assert path.is_file(), "Committed VGG model artifact is missing"
    model = load_model(str(path), compile=False)
    batch = np.zeros((1, 224, 224, 3), dtype=np.float32)
    prediction = np.asarray(model.predict(batch, verbose=0))
    assert prediction.shape == (1, 4), f"Unexpected output shape: {prediction.shape}"
    assert np.all(np.isfinite(prediction)), "Model produced NaN/Inf values"
