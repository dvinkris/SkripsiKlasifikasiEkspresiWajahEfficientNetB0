import os
import sys
import json
import time
import random
import hashlib
import importlib.util
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# LOAD CONFIG
# ============================================================
def load_config():
    current_dir = Path(__file__).resolve().parent
    config_path = current_dir / "09_config.py"

    if not config_path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")

    spec = importlib.util.spec_from_file_location("ferplus_config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    sys.modules["ferplus_config"] = cfg
    spec.loader.exec_module(cfg)

    return cfg


cfg = load_config()


# ============================================================
# BASIC SETUP
# ============================================================
def set_global_seed(seed=cfg.SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
        print("TensorFlow op determinism: ENABLED")
    except Exception as e:
        print("TensorFlow op determinism tidak bisa diaktifkan:", e)


def ensure_directories():
    dirs = [
        cfg.OUTPUT_DIR,
        cfg.MODEL_DIR,
        cfg.LOG_DIR,
        cfg.FIGURE_DIR,
        cfg.RESULT_DIR,
        cfg.GRADCAM_DIR,
        cfg.SNAPSHOT_DIR
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(data, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(json_path):
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_seconds(seconds):
    seconds = float(seconds)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


# ============================================================
# DATASET CHECKING
# ============================================================
def list_image_files(directory):
    directory = Path(directory)

    if not directory.exists():
        return []

    image_files = []

    for file in directory.rglob("*"):
        if file.is_file() and file.suffix.lower() in cfg.VALID_IMAGE_EXTENSIONS:
            image_files.append(file)

    return sorted(image_files)


def is_augmented_file(file_path):
    file_path = Path(file_path)
    return cfg.AUGMENTED_MARKER.lower() in file_path.name.lower()


def check_single_image(image_path):
    image_path = Path(image_path)

    result = {
        "path": str(image_path),
        "filename": image_path.name,
        "extension": image_path.suffix.lower(),
        "is_valid": True,
        "width": None,
        "height": None,
        "mode": None,
        "error": None,
        "is_augmented_filename": is_augmented_file(image_path)
    }

    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            result["width"] = img.size[0]
            result["height"] = img.size[1]
            result["mode"] = img.mode

    except Exception as e:
        result["is_valid"] = False
        result["error"] = str(e)

    return result


def file_md5_hash(file_path, chunk_size=8192):
    file_path = Path(file_path)
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)

    return md5.hexdigest()


def perceptual_hash(image_path, hash_size=8, highfreq_factor=4):
    image_path = Path(image_path)

    img = Image.open(image_path).convert("L")

    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS

    img = img.resize(
        (hash_size * highfreq_factor, hash_size * highfreq_factor),
        resample_method
    )

    pixels = np.asarray(img, dtype=np.float32)

    dct = cv2.dct(pixels)
    dct_low_freq = dct[:hash_size, :hash_size]

    median = np.median(dct_low_freq[1:, 1:])
    diff = dct_low_freq > median

    bit_string = "".join(["1" if v else "0" for v in diff.flatten()])
    hash_hex = hex(int(bit_string, 2))[2:].rjust(hash_size * hash_size // 4, "0")

    return hash_hex


def hamming_distance_hex(hash1, hash2):
    if hash1 is None or hash2 is None:
        return None

    int1 = int(hash1, 16)
    int2 = int(hash2, 16)

    return bin(int1 ^ int2).count("1")


def check_dataset_split(
    split_name,
    split_dir,
    check_corrupt=True,
    compute_hash=True,
    compute_phash=True
):
    split_dir = Path(split_dir)

    print("\n" + "=" * 80)
    print(f"CEK DATASET SPLIT: {split_name}")
    print(f"Path: {split_dir}")
    print("=" * 80)

    if not split_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {split_dir}")

    found_classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

    missing_classes = sorted(list(set(cfg.CLASS_NAMES) - set(found_classes)))
    extra_classes = sorted(list(set(found_classes) - set(cfg.CLASS_NAMES)))

    print("Kelas ditemukan:", found_classes)

    if missing_classes:
        print("PERINGATAN: kelas hilang:", missing_classes)

    if extra_classes:
        print("PERINGATAN: folder kelas tambahan:", extra_classes)

    count_rows = []
    detail_rows = []

    for class_name in cfg.CLASS_NAMES:
        class_dir = split_dir / class_name
        image_files = list_image_files(class_dir)

        valid_count = 0
        corrupt_count = 0
        augmented_count = 0

        widths = []
        heights = []
        modes = []
        extensions = []

        for image_path in image_files:
            info = {
                "split": split_name,
                "class": class_name,
                "path": str(image_path),
                "filename": image_path.name,
                "extension": image_path.suffix.lower(),
                "is_augmented_filename": is_augmented_file(image_path),
                "is_valid": True,
                "width": None,
                "height": None,
                "mode": None,
                "error": None,
                "md5_hash": None,
                "phash": None
            }

            if info["is_augmented_filename"]:
                augmented_count += 1

            if check_corrupt:
                image_info = check_single_image(image_path)
                info.update(image_info)

            if info["is_valid"]:
                valid_count += 1

                widths.append(info["width"])
                heights.append(info["height"])
                modes.append(info["mode"])
                extensions.append(info["extension"])

                if compute_hash:
                    try:
                        info["md5_hash"] = file_md5_hash(image_path)
                    except Exception as e:
                        info["error"] = f"Hash error: {e}"

                if compute_phash:
                    try:
                        info["phash"] = perceptual_hash(image_path)
                    except Exception as e:
                        if info["error"] is None:
                            info["error"] = f"pHash error: {e}"
                        else:
                            info["error"] += f" | pHash error: {e}"

            else:
                corrupt_count += 1

            detail_rows.append(info)

        count_rows.append({
            "split": split_name,
            "class": class_name,
            "total_files": len(image_files),
            "valid_images": valid_count,
            "corrupt_images": corrupt_count,
            "augmented_filename_count": augmented_count,
            "percentage_in_split": None,
            "unique_widths": sorted(list(set(widths))),
            "unique_heights": sorted(list(set(heights))),
            "unique_modes": sorted(list(set(modes))),
            "unique_extensions": sorted(list(set(extensions))),
            "is_empty_class": valid_count == 0
        })

        print(
            f"{class_name:10s} | total: {len(image_files):7d} | "
            f"valid: {valid_count:7d} | corrupt: {corrupt_count:4d} | "
            f"augmented-name: {augmented_count:7d}"
        )

    df_count = pd.DataFrame(count_rows)
    df_detail = pd.DataFrame(detail_rows)

    total_valid = df_count["valid_images"].sum()

    if total_valid > 0:
        df_count["percentage_in_split"] = df_count["valid_images"] / total_valid * 100
    else:
        df_count["percentage_in_split"] = 0

    return df_count, df_detail


def summarize_imbalance(df_count, split_name):
    df = df_count[df_count["split"] == split_name].copy()

    if len(df) == 0:
        raise ValueError(f"Split tidak ditemukan: {split_name}")

    max_row = df.loc[df["valid_images"].idxmax()]
    min_row = df.loc[df["valid_images"].idxmin()]

    max_count = int(max_row["valid_images"])
    min_count = int(min_row["valid_images"])

    if min_count == 0:
        imbalance_ratio = np.inf
    else:
        imbalance_ratio = max_count / min_count

    summary = {
        "split": split_name,
        "total_images": int(df["valid_images"].sum()),
        "max_class": max_row["class"],
        "max_class_count": max_count,
        "min_class": min_row["class"],
        "min_class_count": min_count,
        "imbalance_ratio_max_min": float(imbalance_ratio)
    }

    return summary


def find_exact_duplicates(df_detail):
    df = df_detail.copy()
    df = df[(df["is_valid"] == True) & (df["md5_hash"].notna())]

    duplicate_groups = df.groupby("md5_hash").filter(lambda x: len(x) > 1)

    if duplicate_groups.empty:
        return pd.DataFrame()

    rows = []

    for md5_hash, group in duplicate_groups.groupby("md5_hash"):
        splits = sorted(group["split"].unique().tolist())
        classes = sorted(group["class"].unique().tolist())

        rows.append({
            "md5_hash": md5_hash,
            "duplicate_count": len(group),
            "splits": ", ".join(splits),
            "classes": ", ".join(classes),
            "is_duplicate_across_classes": len(classes) > 1,
            "is_duplicate_across_splits": len(splits) > 1,
            "paths": " | ".join(group["path"].tolist())
        })

    return pd.DataFrame(rows)


def find_near_duplicates_by_phash(df_detail, max_hamming_distance=3, max_pairs=50000):
    df = df_detail.copy()
    df = df[(df["is_valid"] == True) & (df["phash"].notna())].reset_index(drop=True)

    if len(df) < 2:
        return pd.DataFrame()

    rows = []
    checked_pairs = 0

    phash_to_indices = defaultdict(list)

    for idx, ph in enumerate(df["phash"]):
        phash_to_indices[ph].append(idx)

    for ph, indices in phash_to_indices.items():
        if len(indices) > 1:
            group = df.iloc[indices]

            rows.append({
                "phash_1": ph,
                "phash_2": ph,
                "hamming_distance": 0,
                "pair_count_note": f"{len(indices)} images share same phash",
                "splits": ", ".join(sorted(group["split"].unique().tolist())),
                "classes": ", ".join(sorted(group["class"].unique().tolist())),
                "paths": " | ".join(group["path"].head(10).tolist())
            })

    phashes = list(phash_to_indices.keys())

    for i in range(len(phashes)):
        for j in range(i + 1, len(phashes)):
            checked_pairs += 1

            if checked_pairs > max_pairs:
                return pd.DataFrame(rows)

            hdist = hamming_distance_hex(phashes[i], phashes[j])

            if hdist is not None and hdist <= max_hamming_distance:
                idx1 = phash_to_indices[phashes[i]][0]
                idx2 = phash_to_indices[phashes[j]][0]

                row1 = df.iloc[idx1]
                row2 = df.iloc[idx2]

                rows.append({
                    "phash_1": phashes[i],
                    "phash_2": phashes[j],
                    "hamming_distance": hdist,
                    "pair_count_note": "sample pair",
                    "splits": f"{row1['split']} | {row2['split']}",
                    "classes": f"{row1['class']} | {row2['class']}",
                    "paths": f"{row1['path']} | {row2['path']}"
                })

    return pd.DataFrame(rows)


def compare_full_vs_non_augmented(df_count, full_split, non_aug_split):
    full_df = df_count[df_count["split"] == full_split][["class", "valid_images"]].copy()
    non_aug_df = df_count[df_count["split"] == non_aug_split][["class", "valid_images"]].copy()

    full_df = full_df.rename(columns={"valid_images": f"{full_split}_count"})
    non_aug_df = non_aug_df.rename(columns={"valid_images": f"{non_aug_split}_count"})

    merged = pd.merge(full_df, non_aug_df, on="class", how="outer").fillna(0)

    merged["removed_or_difference"] = (
        merged[f"{full_split}_count"] - merged[f"{non_aug_split}_count"]
    )

    merged["remaining_percentage"] = np.where(
        merged[f"{full_split}_count"] > 0,
        merged[f"{non_aug_split}_count"] / merged[f"{full_split}_count"] * 100,
        0
    )

    return merged


# ============================================================
# DATASET LOADER
# ============================================================
def apply_dataset_options(ds):
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.prefetch(1)

    return ds


def make_image_dataset(directory, shuffle, return_file_paths=False):
    directory = Path(directory)

    raw_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(directory),
        labels="inferred",
        label_mode="int",
        class_names=cfg.CLASS_NAMES,
        color_mode=cfg.COLOR_MODE,
        batch_size=cfg.BATCH_SIZE,
        image_size=cfg.IMAGE_SIZE,
        shuffle=shuffle,
        seed=cfg.SEED if shuffle else None
    )

    file_paths = list(getattr(raw_ds, "file_paths", []))

    ds = apply_dataset_options(raw_ds)

    if return_file_paths:
        return ds, file_paths

    return ds


def load_datasets_for_scenario(scenario_key, include_test_paths=False):
    if scenario_key not in cfg.SCENARIO_DATASET_DIRS:
        raise ValueError(f"Skenario tidak dikenali: {scenario_key}")

    dirs = cfg.SCENARIO_DATASET_DIRS[scenario_key]

    train_ds = make_image_dataset(dirs["train"], shuffle=True)
    val_ds = make_image_dataset(dirs["validation"], shuffle=False)

    if include_test_paths:
        test_ds, test_paths = make_image_dataset(
            dirs["test"],
            shuffle=False,
            return_file_paths=True
        )

        return train_ds, val_ds, test_ds, test_paths

    test_ds = make_image_dataset(dirs["test"], shuffle=False)

    return train_ds, val_ds, test_ds


def load_common_test_dataset(return_file_paths=False):
    if return_file_paths:
        return make_image_dataset(
            cfg.TEST_DIR,
            shuffle=False,
            return_file_paths=True
        )

    return make_image_dataset(cfg.TEST_DIR, shuffle=False)


def get_true_labels(dataset):
    y_true = []

    for _, labels in dataset:
        y_true.extend(labels.numpy())

    return np.array(y_true)


# ============================================================
# MODEL BUILDER
# ============================================================
def build_data_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=cfg.SEED),
            layers.RandomRotation(0.02, seed=cfg.SEED),
            layers.RandomTranslation(0.02, 0.02, seed=cfg.SEED),
            layers.RandomZoom(0.03, seed=cfg.SEED),
            layers.RandomContrast(0.03, seed=cfg.SEED)
        ],
        name="data_augmentation"
    )


def build_efficientnetb0_model(
    model_name,
    use_augmentation=False,
    base_trainable=False,
    learning_rate=cfg.LEARNING_RATE_HEAD
):
    tf.keras.backend.clear_session()
    set_global_seed(cfg.SEED)

    inputs = layers.Input(shape=cfg.INPUT_SHAPE, name="input_image")
    x = inputs

    if use_augmentation:
        x = build_data_augmentation()(x)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=cfg.INPUT_SHAPE
    )

    base_model.trainable = base_trainable

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(cfg.DROPOUT_RATE, seed=cfg.SEED, name="dropout")(x)
    outputs = layers.Dense(
        cfg.NUM_CLASSES,
        activation="softmax",
        name="classification_head"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model


def prepare_model_for_fine_tuning(model, last_n_layers=cfg.FINE_TUNE_LAST_N_LAYERS):
    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    for layer in base_model.layers[:-last_n_layers]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE_FINE_TUNE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    return model


def get_callbacks(run_name, checkpoint_path, monitor="val_accuracy"):
    log_path = cfg.LOG_DIR / f"{run_name}_training_log.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.REDUCE_LR_FACTOR,
            patience=cfg.REDUCE_LR_PATIENCE,
            min_lr=cfg.MIN_LR,
            verbose=1
        ),
        CSVLogger(
            filename=str(log_path),
            append=False
        )
    ]

    return callbacks


def plot_training_history(history, title, save_path=None):
    history_dict = history.history

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict.get("accuracy", []), label="train_accuracy")
    plt.plot(history_dict.get("val_accuracy", []), label="val_accuracy")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict.get("loss", []), label="train_loss")
    plt.plot(history_dict.get("val_loss", []), label="val_loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


# ============================================================
# EVALUATION
# ============================================================
def get_model_file_size_mb(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        return None

    return model_path.stat().st_size / (1024 * 1024)


def evaluate_saved_model_on_test(model_label, model_path, test_ds, test_file_paths=None):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    print("\n" + "=" * 80)
    print(f"EVALUASI: {model_label}")
    print(f"Model path: {model_path}")
    print("=" * 80)

    model = tf.keras.models.load_model(model_path)

    eval_start = time.time()
    test_loss, test_accuracy_from_evaluate = model.evaluate(test_ds, verbose=1)
    eval_duration = time.time() - eval_start

    y_true = get_true_labels(test_ds)

    infer_start = time.time()
    y_prob = model.predict(test_ds, verbose=1)
    inference_duration = time.time() - infer_start

    y_pred = np.argmax(y_prob, axis=1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Panjang y_true ({len(y_true)}) dan y_pred ({len(y_pred)}) tidak sama."
        )

    if y_prob.shape[0] != len(y_true):
        raise ValueError(
            f"Jumlah probabilitas ({y_prob.shape[0]}) dan y_true ({len(y_true)}) tidak sama."
        )

    accuracy = accuracy_score(y_true, y_pred)

    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=cfg.CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=cfg.CLASS_NAMES,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    model_size_mb = get_model_file_size_mb(model_path)

    avg_inference_time_per_image_ms = (
        inference_duration / len(y_true) * 1000 if len(y_true) > 0 else None
    )

    summary = {
        "model": model_label,
        "test_loss": float(test_loss),
        "test_accuracy_from_evaluate": float(test_accuracy_from_evaluate),
        "accuracy": float(accuracy),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "num_test_images": int(len(y_true)),
        "evaluation_time_seconds": float(eval_duration),
        "inference_time_seconds": float(inference_duration),
        "avg_inference_time_per_image_ms": float(avg_inference_time_per_image_ms),
        "model_size_mb": float(model_size_mb) if model_size_mb is not None else None,
        "model_path": str(model_path)
    }

    if test_file_paths is None:
        test_file_paths = [""] * len(y_true)

    prediction_df = pd.DataFrame({
        "file_path": test_file_paths,
        "true_label_index": y_true,
        "true_label_name": [cfg.CLASS_NAMES[i] for i in y_true],
        "pred_label_index": y_pred,
        "pred_label_name": [cfg.CLASS_NAMES[i] for i in y_pred],
        "correct": y_true == y_pred
    })

    for idx, class_name in enumerate(cfg.CLASS_NAMES):
        prediction_df[f"prob_{class_name}"] = y_prob[:, idx]

    return summary, report_dict, report_text, cm, prediction_df, y_true, y_pred, y_prob


def plot_confusion_matrix(cm, title, save_path=None):
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(cfg.CLASS_NAMES))
    plt.xticks(tick_marks, cfg.CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, cfg.CLASS_NAMES)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def load_image_as_array(image_path):
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGB")
    img = img.resize(cfg.IMAGE_SIZE)

    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)

    return arr


# ============================================================
# GRAD-CAM
# ============================================================
def call_layer_safely(layer, x, training=False):
    try:
        return layer(x, training=training)
    except TypeError:
        return layer(x)


def find_last_conv_layer_name_in_efficientnet(model):
    base_model = model.get_layer("efficientnetb0")

    last_conv_name = None

    for layer in base_model.layers:
        if isinstance(layer, layers.Conv2D):
            last_conv_name = layer.name

    if last_conv_name is None:
        raise ValueError("Tidak ditemukan Conv2D layer di EfficientNetB0.")

    return last_conv_name


def build_gradcam_model(model, last_conv_layer_name=None):
    base_layer_index = None

    for i, layer in enumerate(model.layers):
        if layer.name == "efficientnetb0":
            base_layer_index = i
            break

    if base_layer_index is None:
        raise ValueError("Layer efficientnetb0 tidak ditemukan pada model.")

    base_model = model.get_layer("efficientnetb0")

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name_in_efficientnet(model)

    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    inputs = model.inputs
    x = inputs[0]

    for layer in model.layers[1:base_layer_index]:
        x = call_layer_safely(layer, x, training=False)

    base_input = x

    backbone_grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    last_conv_output, x = backbone_grad_model(base_input, training=False)

    for layer in model.layers[base_layer_index + 1:]:
        x = call_layer_safely(layer, x, training=False)

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[
            last_conv_output,
            x
        ]
    )

    return grad_model, last_conv_layer_name


def make_gradcam_heatmap(img_array, model, pred_index=None, last_conv_layer_name=None):
    grad_model, used_conv_layer_name = build_gradcam_model(
        model,
        last_conv_layer_name=last_conv_layer_name
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise RuntimeError("Gradient tidak terbentuk. Cek arsitektur model Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if float(max_value.numpy()) > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy(), int(pred_index), predictions.numpy()[0], used_conv_layer_name


def make_gradcam_overlay(image_path, heatmap, alpha=0.4):
    image_path = Path(image_path)

    original_img = cv2.imread(str(image_path))

    if original_img is None:
        raise ValueError(f"Gambar tidak bisa dibaca OpenCV: {image_path}")

    original_img = cv2.resize(original_img, cfg.IMAGE_SIZE)

    heatmap_uint8 = np.uint8(255 * heatmap)
    
    heatmap_uint8 = cv2.resize(
    heatmap_uint8,
    cfg.IMAGE_SIZE
    )
    
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


# ============================================================
# WEBCAM DEMO
# ============================================================
def preprocess_face_crop(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, cfg.IMAGE_SIZE)

    arr = resized.astype("float32")
    arr = np.expand_dims(arr, axis=0)

    return arr


def predict_face_crop_with_models(face_bgr, models_dict):
    input_arr = preprocess_face_crop(face_bgr)

    results = {}

    for model_name, model in models_dict.items():
        probs = model.predict(input_arr, verbose=0)[0]
        pred_index = int(np.argmax(probs))
        pred_class = cfg.CLASS_NAMES[pred_index]
        confidence = float(probs[pred_index])

        results[model_name] = {
            "class": pred_class,
            "confidence": confidence,
            "probs": probs
        }

    return results


def draw_prediction_text(frame, x, y, predictions):
    line_height = 24

    for idx, (model_name, pred_info) in enumerate(predictions.items()):
        text = f"{model_name}: {pred_info['class']} ({pred_info['confidence']:.2f})"

        text_x = x
        text_y = max(25, y - 10 - idx * line_height)

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    return frame