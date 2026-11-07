from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"

TRAIN_FULL_DIR = DATASET_DIR / "training_full"
TRAIN_NON_AUGMENTED_DIR = DATASET_DIR / "training_non_augmented"

VALIDATION_FULL_DIR = DATASET_DIR / "validation_full"
VALIDATION_NON_AUGMENTED_DIR = DATASET_DIR / "validation_non_augmented"

TEST_DIR = DATASET_DIR / "test"


CLASS_NAMES = [
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

NUM_CLASSES = len(CLASS_NAMES)


ORIGINAL_IMAGE_SIZE = (112, 112)
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
COLOR_MODE = "rgb"


AUGMENTED_MARKER = "augmented"


SEED = 42


BATCH_SIZE = 32

MAX_EPOCHS_HEAD = 30


# Untuk Model 3:
EPOCHS_MODEL_3_HEAD = 30
EPOCHS_MODEL_3_FINE_TUNE = 30

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

DROPOUT_RATE = 0.3

FINE_TUNE_LAST_N_LAYERS = 50


EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LR = 1e-7


OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
SNAPSHOT_DIR = OUTPUT_DIR / "webcam_snapshots"


MODEL_1_NAME = "model_1_baseline_no_augmentation"
MODEL_2_NAME = "model_2_training_full_with_augmentation"
MODEL_3_PHASE_1_NAME = "model_3_phase_1_head_training"
MODEL_3_NAME = "model_3_training_full_augmentation_finetuning"


MODEL_1_BEST_PATH = MODEL_DIR / f"{MODEL_1_NAME}_best.keras"
MODEL_1_FINAL_PATH = MODEL_DIR / f"{MODEL_1_NAME}_final.keras"

MODEL_2_BEST_PATH = MODEL_DIR / f"{MODEL_2_NAME}_best.keras"
MODEL_2_FINAL_PATH = MODEL_DIR / f"{MODEL_2_NAME}_final.keras"

MODEL_3_PHASE_1_BEST_PATH = MODEL_DIR / f"{MODEL_3_PHASE_1_NAME}_best.keras"

MODEL_3_BEST_PATH = MODEL_DIR / f"{MODEL_3_NAME}_best.keras"
MODEL_3_FINAL_PATH = MODEL_DIR / f"{MODEL_3_NAME}_final.keras"

MODEL_PATHS = {
    "Model 1 - Baseline No Augmentation": MODEL_1_BEST_PATH,
    "Model 2 - Training Full + Augmentation": MODEL_2_BEST_PATH,
    "Model 3 - Training Full + Augmentation + Fine Tuning": MODEL_3_BEST_PATH
}


SCENARIO_DATASET_DIRS = {
    "model_1": {
        "train": TRAIN_NON_AUGMENTED_DIR,
        "validation": VALIDATION_NON_AUGMENTED_DIR,
        "test": TEST_DIR
    },
    "model_2": {
        "train": TRAIN_FULL_DIR,
        "validation": VALIDATION_NON_AUGMENTED_DIR,
        "test": TEST_DIR
    },
    "model_3": {
        "train": TRAIN_FULL_DIR,
        "validation": VALIDATION_NON_AUGMENTED_DIR,
        "test": TEST_DIR
    }
}


DATASET_SPLITS_TO_CHECK = {
    "training_full": TRAIN_FULL_DIR,
    "training_non_augmented": TRAIN_NON_AUGMENTED_DIR,
    "validation_full": VALIDATION_FULL_DIR,
    "validation_non_augmented": VALIDATION_NON_AUGMENTED_DIR,
    "test": TEST_DIR
}

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif"
}


SCENARIO_NOTES = {
    "model_1": (
        "Model 1 menggunakan training_non_augmented sebagai baseline. "
        "Tidak menggunakan augmentation tambahan, tidak menggunakan class_weight, "
        "tidak menggunakan oversampling/undersampling, dan tidak menggunakan fine-tuning."
    ),
    "model_2": (
        "Model 2 menggunakan training_full, yaitu data training Kaggle yang masih berisi "
        "gambar asli dan gambar augmented bawaan dataset. Model ini juga menggunakan "
        "augmentation tambahan melalui pipeline training. Tidak menggunakan fine-tuning."
    ),
    "model_3": (
        "Model 3 menggunakan training_full yang sama persis dengan Model 2, augmentation "
        "tambahan yang sama persis dengan Model 2, lalu dilanjutkan dengan fine-tuning "
        "sebagian layer atas EfficientNetB0."
    )
}