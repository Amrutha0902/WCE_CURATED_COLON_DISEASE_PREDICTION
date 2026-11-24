"""Evaluate saved VGG on an independent, labeled WCE test manifest.

Manifest CSV columns: image_path,label,patient_id. Relative paths are resolved
against --image-root. Never use training/validation images for this report.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

CLASSES = ["Normal", "Ulcerative Colitis", "Polyps", "Esophagitis"]


def metrics(truth, predicted):
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    for y, p in zip(truth, predicted):
        confusion[y][p] += 1
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted_total = sum(row[i] for row in confusion)
        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return {
        "count": len(truth),
        "accuracy": sum(y == p for y, p in zip(truth, predicted)) / len(truth),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / len(CLASSES),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "class_order": CLASSES,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("model_1/Vgg.h5"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_report.json"))
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error("A labeled, independent test manifest is required; no scores will be fabricated.")
    with args.manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"image_path", "label", "patient_id"}.issubset(reader.fieldnames or []):
            parser.error("Manifest requires image_path,label,patient_id")
        records = list(reader)
    if not records:
        parser.error("No evaluation rows")
    if any(not row["patient_id"].strip() for row in records):
        parser.error("Every sample requires a patient_id for leakage auditing")
    if any(row["label"] not in CLASSES for row in records):
        parser.error("Unknown class label; check model class order against training")
    if len({row["image_path"] for row in records}) != len(records):
        parser.error("Duplicate image paths in evaluation manifest")
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"
    from keras.models import load_model
    from keras.preprocessing.image import load_img, img_to_array

    model = load_model(str(args.model), compile=False)
    y_true, y_pred = [], []
    root = args.image_root.resolve()
    for row in records:
        image_path = (root / row["image_path"]).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            parser.error("Image path outside root or missing: " + row["image_path"])
        image = img_to_array(load_img(image_path, target_size=(224, 224)))
        prediction = np.asarray(model.predict(np.expand_dims(image / 255.0, 0), verbose=0))
        if prediction.shape != (1, len(CLASSES)) or not np.all(np.isfinite(prediction)):
            raise ValueError("Unexpected model output")
        y_true.append(CLASSES.index(row["label"]))
        y_pred.append(int(np.argmax(prediction[0])))
    result = metrics(y_true, y_pred)
    result["unique_patients"] = len({row["patient_id"] for row in records})
    result["class_distribution"] = dict(Counter(row["label"] for row in records))
    result["limitations"] = (
        "Patient IDs enable auditing but do not prove train/test patient disjointness. "
        "Compare these IDs against the training manifest before citing results. "
        "Class ordering must be confirmed from the original training notebook."
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
