import sys
import tempfile
import importlib.util
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QFrame,
    QSizePolicy,
    QScrollArea,
    QLayout,
)

# ============================================================
# PROJECT ROOT + IMPORT 08 / 09
# ============================================================
def find_project_root(start_path=None):
    current = Path(start_path or Path(__file__).resolve().parent).resolve()

    while True:
        if (current / "08_utils.py").exists() and (current / "09_config.py").exists():
            return current

        if current.parent == current:
            raise FileNotFoundError(
                "Project root tidak ditemukan. "
                "Pastikan 08_utils.py dan 09_config.py tersedia."
            )

        current = current.parent


def import_py_file(module_name, file_path):
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, file_path)

    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


PROJECT_ROOT = find_project_root()

cfg = import_py_file("ferplus_config_desktop", PROJECT_ROOT / "09_config.py")

utils = import_py_file("ferplus_utils_desktop", PROJECT_ROOT / "08_utils.py")

utils.set_global_seed(cfg.SEED)
utils.ensure_directories()


# ============================================================
# UI-ONLY CONSTANTS
# ============================================================
MODEL_PATHS_UI = {
    "M1 Baseline": cfg.MODEL_1_BEST_PATH,
    "M2 Aug": cfg.MODEL_2_BEST_PATH,
    "M3 Aug+FT": cfg.MODEL_3_BEST_PATH,
}

MODEL_3_KEY = "M3 Aug+FT"

GRADCAM_ALPHA = 0.4

MODEL_LABELS_UI = {
    "M1 Baseline": "Model 1 — Baseline",
    "M2 Aug": "Model 2 — Augmentasi",
    "M3 Aug+FT": "Model 3 — Augmentasi + FT",
}


# ============================================================
# LOAD MODELS
# ============================================================
def load_models():
    models_dict = {}

    for model_name, model_path in MODEL_PATHS_UI.items():
        model_path = Path(model_path)

        print(f"Loading {model_name}: " f"{model_path}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

        models_dict[model_name] = tf.keras.models.load_model(model_path)

        print(f"{model_name} loaded.")

    return models_dict


# ============================================================
# FACE DETECTOR
# ============================================================
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Haar Cascade face detector gagal dimuat.")

    return detector


# ============================================================
# IMAGE LABEL
# ============================================================
class ImageLabel(QLabel):

    def __init__(
        self,
        text="",
        preferred_size=(480, 360),
        minimum_size=(120, 90),
        fill_area=False,
    ):
        super().__init__(text)
        self.original_pixmap = None
        self.fill_area = fill_area

        self._preferred_size = QSize(*preferred_size)
        self._minimum_size = QSize(*minimum_size)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("imagePanel")
        self.setWordWrap(True)
        self.setMinimumSize(self._minimum_size)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
    
    def sizeHint(self):
        return self._preferred_size

    def minimumSizeHint(self):
        return self._minimum_size

    def set_bgr_image(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return
        conversion = cv2.COLOR_GRAY2RGB if image_bgr.ndim == 2 else cv2.COLOR_BGR2RGB
        image_rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, conversion))
        height, width, _ = image_rgb.shape
        q_image = QImage(
            image_rgb.data,
            width,
            height,
            image_rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()
        self.original_pixmap = QPixmap.fromImage(q_image)
        self._refresh_pixmap()

    def clear_image(self, text=""):
        self.original_pixmap = None
        self.clear()
        self.setText(text)

    def _refresh_pixmap(self):
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return

        target_size = self.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        aspect_mode = (
            Qt.AspectRatioMode.KeepAspectRatioByExpanding
            if self.fill_area
            else Qt.AspectRatioMode.KeepAspectRatio
        )

        scaled = self.original_pixmap.scaled(
            target_size,
            aspect_mode,
            Qt.TransformationMode.SmoothTransformation,
        )

        if self.fill_area:
            left = max(
                0, (scaled.width() - target_size.width()) // 2
            )
            top = max(
                0, (scaled.height() - target_size.height()) // 2
            )

            scaled = scaled.copy(
                left,
                top,
                target_size.width(),
                target_size.height(),
            )

        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()


# ============================================================
# PREDICTION CARD
# ============================================================
class PredictionCard(QFrame):

    def __init__(self, model_name, final=False):
        super().__init__()
        self.setObjectName("finalPredictionCard" if final else "predictionCard")
        self.setMinimumWidth(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.card_layout = QVBoxLayout(self)
        self.card_layout.setContentsMargins(10, 6, 10, 6)
        self.card_layout.setSpacing(2)

        self.model_label = QLabel(model_name)
        self.model_label.setObjectName("modelName")

        self.prediction_label = QLabel("-")
        self.prediction_label.setObjectName("predictionValue")
        self.prediction_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )

        self.confidence_label = QLabel("Confidence: -")
        self.confidence_label.setObjectName("mutedText")
        self.confidence_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )

        result_row = QHBoxLayout()
        result_row.setSpacing(8)
        result_row.addWidget(self.prediction_label, 1)
        result_row.addWidget(self.confidence_label)
        self.card_layout.addWidget(self.model_label)
        self.card_layout.addLayout(result_row)

    def set_compact(self, compact):
        vertical_margin = 4 if compact else 6
        self.card_layout.setContentsMargins(10, vertical_margin, 10, vertical_margin)
        self.card_layout.setSpacing(1 if compact else 2)
        self.setMinimumHeight(46 if compact else 56)

    def update_prediction(self, pred_info):
        if pred_info is None:
            self.prediction_label.setText("-")
            self.confidence_label.setText("Confidence: -")
            return
        self.prediction_label.setText(pred_info["class"].upper())
        self.confidence_label.setText(
            f"Confidence: {pred_info['confidence'] * 100:.2f}%"
        )


# ============================================================
# WEBCAM WORKER
# ============================================================
class WebcamWorker(QThread):

    frame_ready = Signal(object, object, object)

    error_occurred = Signal(str)

    camera_status = Signal(str)

    def __init__(self, models_dict, camera_index=0):
        super().__init__()

        self.models_dict = models_dict

        self.camera_index = camera_index

    def run(self):

        cap = None

        try:
            face_detector = load_face_detector()


            if sys.platform.startswith("win"):

                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

            else:

                cap = cv2.VideoCapture(self.camera_index)

            if not cap.isOpened():
                raise RuntimeError("Webcam tidak dapat dibuka.")

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            except Exception:
                pass

            self.camera_status.emit("Webcam aktif.")

            while not self.isInterruptionRequested():

                ret, frame = cap.read()

                if not ret:
                    self.error_occurred.emit("Frame webcam tidak terbaca.")

                    break

                display_frame = frame.copy()

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_detector.detectMultiScale(
                    gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
                )

                current_face_crop = None

                current_predictions = None

                largest_area = 0

                largest_box = None


                for x, y, w, h in faces:

                    face_crop = frame[y : y + h, x : x + w]

                    if face_crop.size == 0:
                        continue


                    predictions = utils.predict_face_crop_with_models(
                        face_bgr=face_crop, models_dict=self.models_dict
                    )

                    face_area = w * h

                    if face_area > largest_area:

                        largest_area = face_area

                        current_face_crop = face_crop.copy()

                        current_predictions = predictions

                        largest_box = (x, y, w, h)


                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)


                    display_frame = utils.draw_prediction_text(
                        frame=display_frame, x=x, y=y, predictions=predictions
                    )


                    if MODEL_3_KEY in predictions:

                        final_pred = predictions[MODEL_3_KEY]

                        final_text = (
                            f"Final M3: "
                            f"{final_pred['class']} "
                            f"("
                            f"{final_pred['confidence']:.2f}"
                            f")"
                        )

                        cv2.putText(
                            display_frame,
                            final_text,
                            (x, min(display_frame.shape[0] - 10, y + h + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )


                if largest_box is not None:

                    x, y, w, h = largest_box

                    cv2.rectangle(
                        display_frame, (x, y), (x + w, y + h), (0, 0, 255), 3
                    )


                self.frame_ready.emit(
                    display_frame.copy(),
                    (
                        current_face_crop.copy()
                        if current_face_crop is not None
                        else None
                    ),
                    current_predictions,
                )

                self.msleep(1)

        except Exception as exc:

            self.error_occurred.emit(str(exc))

        finally:

            if cap is not None:
                cap.release()

            self.camera_status.emit("Webcam berhenti.")


# ============================================================
# DISPLAY HELPER
# ============================================================
def add_title_to_image(image_bgr, title_lines):

    image_bgr = cv2.resize(image_bgr, cfg.IMAGE_SIZE)

    title_height = 78

    title_area = np.full((title_height, image_bgr.shape[1], 3), 245, dtype=np.uint8)

    y = 22

    for line in title_lines:

        cv2.putText(
            title_area,
            str(line),
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )

        y += 22

    return cv2.vconcat([title_area, image_bgr])


# ============================================================
# GRAD-CAM ANALYSIS
# ============================================================
def create_gradcam_analysis(
    face_crop_bgr, annotated_frame_bgr, models_dict, trigger, source
):

    snapshot_time = datetime.now()

    input_arr = utils.preprocess_face_crop(face_crop_bgr)

    input_rgb = input_arr[0].astype("uint8")

    input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)

    face_gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

    face_gray_224 = cv2.resize(face_gray, cfg.IMAGE_SIZE)

    predictions = {}

    overlays = {}

    used_layers = {}


    with tempfile.TemporaryDirectory() as temp_dir:

        temp_input_path = Path(temp_dir) / "gradcam_input.png"

        cv2.imwrite(str(temp_input_path), input_bgr)


        for model_name, model in models_dict.items():

            heatmap, pred_index, probs, used_conv_layer = utils.make_gradcam_heatmap(
                img_array=input_arr,
                model=model,
                pred_index=None,
                last_conv_layer_name=None,
            )

            overlay_bgr = utils.make_gradcam_overlay(
                image_path=temp_input_path, heatmap=heatmap, alpha=GRADCAM_ALPHA
            )

            predictions[model_name] = {
                "class": cfg.CLASS_NAMES[pred_index],
                "confidence": float(probs[pred_index]),
                "probs": probs,
            }

            overlays[model_name] = overlay_bgr

            used_layers[model_name] = used_conv_layer


    panels = [
        add_title_to_image(face_crop_bgr, ["Wajah Asli", "Face crop"]),
        add_title_to_image(input_bgr, ["Input Model", "224x224x3 grayscale"]),
    ]

    for model_name in MODEL_PATHS_UI.keys():

        pred = predictions[model_name]

        panels.append(
            add_title_to_image(
                overlays[model_name],
                [
                    model_name,
                    (f"{pred['class']} " f"(" f"{pred['confidence']:.2f}" f")"),
                    (f"Layer: " f"{used_layers[model_name]}"),
                ],
            )
        )

    gradcam_comparison = cv2.hconcat(panels)

    return {
        "snapshot_time": snapshot_time,
        "finished_time": datetime.now(),
        "trigger": trigger,
        "source": source,
        "annotated_frame_bgr": annotated_frame_bgr.copy(),
        "face_crop_bgr": face_crop_bgr.copy(),
        "face_gray_224": face_gray_224,
        "input_bgr": input_bgr,
        "predictions": predictions,
        "overlays": overlays,
        "used_layers": used_layers,
        "gradcam_comparison_bgr": gradcam_comparison,
    }


# ============================================================
# GRAD-CAM WORKER
# ============================================================
class GradCAMWorker(QThread):

    analysis_ready = Signal(object)

    error_occurred = Signal(str)

    def __init__(
        self, face_crop_bgr, annotated_frame_bgr, models_dict, trigger, source
    ):
        super().__init__()

        self.face_crop_bgr = face_crop_bgr.copy()

        self.annotated_frame_bgr = annotated_frame_bgr.copy()

        self.models_dict = models_dict

        self.trigger = trigger

        self.source = source

    def run(self):

        try:

            analysis = create_gradcam_analysis(
                face_crop_bgr=self.face_crop_bgr,
                annotated_frame_bgr=self.annotated_frame_bgr,
                models_dict=self.models_dict,
                trigger=self.trigger,
                source=self.source,
            )

            self.analysis_ready.emit(analysis)

        except Exception as exc:

            self.error_occurred.emit(str(exc))


# ============================================================
# MAIN WINDOW
# ============================================================
class FERDesktopApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.models_dict = load_models()
        self.webcam_worker = None
        self.gradcam_worker = None
        self.current_frame = None
        self.current_face_crop = None
        self.current_predictions = None
        self.current_source = "idle"
        self.latest_analysis = None
        self.auto_gradcam_timer = QTimer(self)
        self.auto_gradcam_timer.timeout.connect(self.request_auto_gradcam)
        self.setWindowTitle("FERPlus EfficientNetB0 - Klasifikasi Ekspresi Wajah")
        self._compact_ui = None
        self._controls_stacked = None
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.setMinimumSize(
                min(760, available.width()), min(480, available.height())
            )
            self.resize(
                min(1280, max(760, available.width() - 40)),
                min(800, max(480, available.height() - 60)),
            )
        else:
            self.setMinimumSize(760, 480)
            self.resize(1200, 720)
        self.build_ui()
        self._update_responsive_layout()
        self.theme_combo.setCurrentText("Light")
        self.apply_theme("Light")
        self.gradcam_interval_combo.setCurrentText("Manual")
        self.update_auto_gradcam_timer()


    def build_ui(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.setCentralWidget(self.scroll_area)

        self.page = QWidget()
        self.page.setObjectName("page")
        self.page.setMinimumWidth(760)
        self.scroll_area.setWidget(self.page)

        self.root_layout = QVBoxLayout(self.page)
        self.root_layout.setContentsMargins(10, 10, 10, 10)
        self.root_layout.setSpacing(8)
        self.root_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        header_layout = QHBoxLayout()
        title_layout = QVBoxLayout()
        title_layout.setSpacing(0)
        self.title_label = QLabel("Klasifikasi Ekspresi Wajah")
        self.title_label.setObjectName("titleLabel")
        subtitle = QLabel("Transfer Learning EfficientNetB0 pada Dataset FERPlus")
        subtitle.setObjectName("subtitleLabel")
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout, 1)
        header_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light 🔆", "Dark 🌙"])
        self.theme_combo.setMinimumWidth(88)
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        header_layout.addWidget(self.theme_combo)
        self.root_layout.addLayout(header_layout)

        self.control_frame = QFrame()
        self.control_frame.setObjectName("card")
        self.control_layout = QGridLayout(self.control_frame)
        self.control_layout.setContentsMargins(8, 5, 8, 5)
        self.control_layout.setSpacing(6)
        self.button_bar = QWidget()
        button_layout = QHBoxLayout(self.button_bar)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)

        self.start_camera_button = QPushButton("▶️ Start Webcam")
        self.stop_camera_button = QPushButton("🛑 Stop Webcam")
        self.photo_button = QPushButton("🖼️ Pilih Foto")
        self.gradcam_button = QPushButton("🌈 Grad-CAM Captured")
        self.save_button = QPushButton("📥 Simpan File Hasil Klasifikasi")
        self.gradcam_button.setToolTip("Tampilkan Grad-CAM untuk wajah saat ini")
        self.save_button.setToolTip(
            "Simpan lima file dari analisis Grad-CAM terakhir"
        )
        self.stop_camera_button.setEnabled(False)
        self.start_camera_button.clicked.connect(self.start_webcam)
        self.stop_camera_button.clicked.connect(self.stop_webcam)
        self.photo_button.clicked.connect(self.select_photo)
        self.gradcam_button.clicked.connect(self.request_manual_gradcam)
        self.save_button.clicked.connect(self.save_latest_analysis)
        for button in (
            self.start_camera_button,
            self.stop_camera_button,
            self.photo_button,
            self.gradcam_button,
            self.save_button,
        ):
            button_layout.addWidget(button)
        button_layout.addStretch(1)

        self.interval_bar = QWidget()
        interval_layout = QHBoxLayout(self.interval_bar)
        interval_layout.setContentsMargins(0, 0, 0, 0)
        interval_layout.setSpacing(6)
        interval_layout.addWidget(QLabel("Auto Grad-CAM:"))
        self.gradcam_interval_combo = QComboBox()
        self.gradcam_interval_combo.addItems(
            ["Manual", "5 seconds", "8 seconds (recommended)", "10 seconds"]
        )
        self.gradcam_interval_combo.setMinimumWidth(150)
        self.gradcam_interval_combo.currentTextChanged.connect(
            self.update_auto_gradcam_timer
        )
        interval_layout.addWidget(self.gradcam_interval_combo)
        self.control_layout.addWidget(self.button_bar, 0, 0)
        self.control_layout.addWidget(self.interval_bar, 0, 1)
        self.control_layout.setColumnStretch(0, 1)
        self.root_layout.addWidget(self.control_frame)


        self.top_container = QWidget()
        self.top_layout = QHBoxLayout(self.top_container)
        self.top_layout.setContentsMargins(0, 0, 0, 0)
        self.top_layout.setSpacing(10)

        self.video_frame = QFrame()
        self.video_frame.setObjectName("card")
        self.video_layout = QVBoxLayout(self.video_frame)
        self.video_layout.setContentsMargins(8, 7, 8, 8)
        self.video_layout.setSpacing(5)
        video_title = QLabel("Webcam / Foto Input")
        video_title.setObjectName("sectionTitle")

        self.video_label = ImageLabel(
            "Klik 'Start Webcam' atau Pilih foto.",
            preferred_size=(640, 480),
            minimum_size=(200, 120),
            #fill_area=True, #Untuk tampilan webcam/foto full 1 area panel
            fill_area=False, #Untuk tampilan webcam/foto supaya utuh tidak terpotong
        )
        self.video_layout.addWidget(video_title)
        self.video_layout.addWidget(self.video_label, 1)
        self.top_layout.addWidget(self.video_frame, 3)

        self.prediction_frame = QFrame()
        self.prediction_frame.setObjectName("card")
        self.prediction_layout = QVBoxLayout(self.prediction_frame)
        self.prediction_layout.setContentsMargins(8, 7, 8, 8)
        self.prediction_layout.setSpacing(6)
        prediction_title = QLabel("Prediksi Ekspresi")
        prediction_title.setObjectName("sectionTitle")
        self.prediction_layout.addWidget(prediction_title)
        self.model_cards = {}
        for model_name in MODEL_PATHS_UI:
            card = PredictionCard(MODEL_LABELS_UI[model_name])
            self.model_cards[model_name] = card
            self.prediction_layout.addWidget(card, 1)
        self.final_card = PredictionCard("HASIL AKHIR — MODEL 3", final=True)
        self.prediction_layout.addWidget(self.final_card, 1)
        self.top_layout.addWidget(self.prediction_frame, 2)
        self.root_layout.addWidget(self.top_container, 4)


        self.gradcam_frame = QFrame()
        self.gradcam_frame.setObjectName("card")
        self.gradcam_layout = QVBoxLayout(self.gradcam_frame)
        self.gradcam_layout.setContentsMargins(8, 7, 8, 8)
        self.gradcam_layout.setSpacing(5)
        gradcam_header = QHBoxLayout()
        gradcam_title = QLabel("Visualisasi Grad-CAM")
        gradcam_title.setObjectName("sectionTitle")
        self.gradcam_status_label = QLabel("Belum ada Grad-CAM.")
        self.gradcam_status_label.setObjectName("mutedText")
        self.gradcam_status_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.gradcam_status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        gradcam_header.addWidget(gradcam_title)
        gradcam_header.addWidget(self.gradcam_status_label, 1)
        self.gradcam_layout.addLayout(gradcam_header)

        self.gradcam_grid = QGridLayout()
        self.gradcam_grid.setContentsMargins(0, 0, 0, 0)
        self.gradcam_grid.setHorizontalSpacing(8)
        self.gradcam_labels = {}
        self.gradcam_caption_labels = {}
        self.gradcam_panels = {}
        for column, model_name in enumerate(MODEL_PATHS_UI):
            panel = QFrame()
            panel.setObjectName("gradcamCard")
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(6, 5, 6, 5)
            panel_layout.setSpacing(3)
            caption = QLabel(f"{MODEL_LABELS_UI[model_name]}\nMenunggu analisis")
            caption.setObjectName("gradcamCaption")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption.setWordWrap(True)
            image_label = ImageLabel(
                "Menunggu Grad-CAM...",
                preferred_size=(224, 224),
                minimum_size=(72, 72),
            )
            panel_layout.addWidget(caption)
            panel_layout.addWidget(image_label, 1)
            self.gradcam_labels[model_name] = image_label
            self.gradcam_caption_labels[model_name] = caption
            self.gradcam_panels[model_name] = panel
            self.gradcam_grid.addWidget(panel, 0, column)
            self.gradcam_grid.setColumnStretch(column, 1)
        self.gradcam_grid.setRowStretch(0, 1)
        self.gradcam_layout.addLayout(self.gradcam_grid, 1)
        self.root_layout.addWidget(self.gradcam_frame, 2)

        self.status_label = QLabel("Siap.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.root_layout.addWidget(self.status_label)


    def apply_theme(self, theme):
        if theme == "Dark 🌙":
            palette = {
                "background": "#0D1117",
                "text": "#F3F6FA",
                "card": "#151B23",
                "border": "#2B3440",
                "inner": "#1B2430",
                "inner_border": "#334155",
                "muted": "#AEBBC9",
                "image": "#080B0F",
                "final": "#102B3A",
                "accent": "#38BDF8",
                "button": "#263241",
                "hover": "#334155",
                "pressed": "#1E293B",
                "disabled": "#171D26",
                "disabled_text": "#667085",
            }
        else:
            palette = {
                "background": "#F5F7FA",
                "text": "#172033",
                "card": "#FFFFFF",
                "border": "#D9E1EA",
                "inner": "#F2F5F9",
                "inner_border": "#D6DEE8",
                "muted": "#5F6B7A",
                "image": "#E9EEF4",
                "final": "#EAF4FF",
                "accent": "#1976D2",
                "button": "#FFFFFF",
                "hover": "#EAF2FB",
                "pressed": "#DCE9F7",
                "disabled": "#ECEFF3",
                "disabled_text": "#9CA6B3",
            }
        compact = self._compact_ui
        body_size = 11 if compact else 12
        title_size = 19 if compact else 23
        section_size = 13 if compact else 14
        prediction_size = 18 if compact else 22
        button_padding = 4 if compact else 6
        p = palette
        self.setStyleSheet(f"""
            QWidget {{
                color: {p['text']};
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: {body_size}px;
            }}
            QMainWindow, QWidget#page, QScrollArea {{
                background-color: {p['background']};
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
            QFrame#card {{
                background-color: {p['card']};
                border: 1px solid {p['border']};
                border-radius: 10px;
            }}
            QFrame#predictionCard, QFrame#gradcamCard {{
                background-color: {p['inner']};
                border: 1px solid {p['inner_border']};
                border-radius: 8px;
            }}
            QFrame#finalPredictionCard {{
                background-color: {p['final']};
                border: 1px solid {p['accent']};
                border-radius: 8px;
            }}
            QLabel#titleLabel {{
                font-size: {title_size}px;
                font-weight: 700;
            }}
            QLabel#sectionTitle {{
                font-size: {section_size}px;
                font-weight: 700;
            }}
            QLabel#modelName, QLabel#gradcamCaption {{
                font-weight: 600;
            }}
            QLabel#predictionValue {{
                font-size: {prediction_size}px;
                font-weight: 700;
            }}
            QLabel#subtitleLabel, QLabel#mutedText, QLabel#statusLabel {{
                color: {p['muted']};
            }}
            QLabel#imagePanel {{
                background-color: {p['image']};
                border: 1px solid {p['inner_border']};
                border-radius: 6px;
                color: {p['muted']};
            }}
            QPushButton {{
                background-color: {p['button']};
                border: 1px solid {p['inner_border']};
                border-radius: 6px;
                padding: {button_padding}px 9px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {p['hover']};
            }}
            QPushButton:pressed {{
                background-color: {p['pressed']};
            }}
            QPushButton:disabled {{
                background-color: {p['disabled']};
                color: {p['disabled_text']};
            }}
            QComboBox {{
                background-color: {p['card']};
                border: 1px solid {p['inner_border']};
                border-radius: 6px;
                padding: {button_padding}px 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {p['card']};
                selection-background-color: {p['hover']};
            }}
            QMessageBox {{
                 background-color: {p['card']};
            }}
            QMessageBox QLabel {{
                color: {p['text']};
                background-color: transparent;
            }}
            QMessageBox QPushButton {{
                min-width: 90px;
            }}
            """)

    def _update_responsive_layout(self):
        """Ukuran dalam logical pixel; Qt menangani scaling DPI layar."""
        if not hasattr(self, "root_layout"):
            return
        compact = self.height() < 660 or self.width() < 980
        stacked = self.width() < 960
        if compact != self._compact_ui:
            self._compact_ui = compact
            margin = 8 if compact else 10
            gap = 6 if compact else 8
            self.root_layout.setContentsMargins(margin, margin, margin, margin)
            self.root_layout.setSpacing(gap)
            self.top_layout.setSpacing(gap)
            self.prediction_layout.setSpacing(4 if compact else 6)
            for layout in (
                self.video_layout,
                self.prediction_layout,
                self.gradcam_layout,
            ):
                layout.setContentsMargins(
                    7, 5 if compact else 7, 7, 6 if compact else 8
                )
            for card in [*self.model_cards.values(), self.final_card]:
                card.set_compact(compact)
            self.apply_theme(self.theme_combo.currentText())

        if stacked != self._controls_stacked:
            self._controls_stacked = stacked
            self.control_layout.removeWidget(self.button_bar)
            self.control_layout.removeWidget(self.interval_bar)
            if stacked:
                self.control_layout.addWidget(self.button_bar, 0, 0, 1, 2)
                self.control_layout.addWidget(
                    self.interval_bar, 1, 0, 1, 2, Qt.AlignmentFlag.AlignRight
                )
            else:
                self.control_layout.addWidget(self.button_bar, 0, 0)
                self.control_layout.addWidget(self.interval_bar, 0, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()


    def start_webcam(self):

        if self.webcam_worker is not None and self.webcam_worker.isRunning():
            return

        self.current_source = "webcam"

        self.current_frame = None

        self.current_face_crop = None

        self.current_predictions = None

        self.update_prediction_cards(None)

        self.webcam_worker = WebcamWorker(self.models_dict, camera_index=0)

        self.webcam_worker.frame_ready.connect(self.on_webcam_frame)

        self.webcam_worker.error_occurred.connect(self.on_worker_error)

        self.webcam_worker.camera_status.connect(self.status_label.setText)

        self.webcam_worker.finished.connect(self.on_webcam_finished)

        self.webcam_worker.start()

        self.start_camera_button.setEnabled(False)

        self.stop_camera_button.setEnabled(True)

        self.update_auto_gradcam_timer()


    def stop_webcam(self):

        self.auto_gradcam_timer.stop()

        if self.webcam_worker is not None and self.webcam_worker.isRunning():

            self.webcam_worker.requestInterruption()

            self.webcam_worker.wait()

        self.webcam_worker = None

        self.start_camera_button.setEnabled(True)

        self.stop_camera_button.setEnabled(False)

        self.status_label.setText("Webcam berhenti.")

    def on_webcam_finished(self):

        self.auto_gradcam_timer.stop()

        self.start_camera_button.setEnabled(True)

        self.stop_camera_button.setEnabled(False)


    def on_webcam_frame(self, frame_bgr, face_crop, predictions):

        if self.current_source != "webcam":
            return

        self.current_frame = frame_bgr.copy()

        self.current_face_crop = (
            face_crop.copy()
            if face_crop is not None
            else None
        )

        self.current_predictions = predictions

        self.video_label.set_bgr_image(frame_bgr)

        self.update_prediction_cards(predictions)


    def update_prediction_cards(self, predictions):

        if predictions is None:

            for card in self.model_cards.values():

                card.update_prediction(None)

            self.final_card.update_prediction(None)

            return

        for model_name, card in self.model_cards.items():

            card.update_prediction(predictions.get(model_name))

        self.final_card.update_prediction(predictions.get(MODEL_3_KEY))


    def get_gradcam_interval_ms(self):

        text = self.gradcam_interval_combo.currentText()

        if text.startswith("5 "):
            return 5000

        if text.startswith("8 "):
            return 8000

        if text.startswith("10 "):
            return 10000

        return 0

    def update_auto_gradcam_timer(self):

        interval_ms = self.get_gradcam_interval_ms()

        webcam_active = (
            self.webcam_worker is not None and self.webcam_worker.isRunning()
        )

        if interval_ms > 0 and webcam_active:

            self.auto_gradcam_timer.start(interval_ms)

        else:

            self.auto_gradcam_timer.stop()

    def request_auto_gradcam(self):

        if self.gradcam_worker is not None and self.gradcam_worker.isRunning():
            return

        if self.current_face_crop is None:
            return

        if self.current_frame is None:
            return

        self.start_gradcam_analysis(trigger="auto", source="webcam")


    def request_manual_gradcam(self):

        if self.current_face_crop is None or self.current_frame is None:

            QMessageBox.warning(
                self,
                "Grad-CAM",
                ("Tidak ada wajah yang sedang " "tersedia untuk dianalisis."),
            )

            return

        if self.gradcam_worker is not None and self.gradcam_worker.isRunning():

            self.status_label.setText("Grad-CAM masih diproses. " "Tunggu sebentar.")

            return

        source = "photo" if self.current_source == "photo" else "webcam"

        self.start_gradcam_analysis(trigger="manual", source=source)

        if source == "webcam" and self.auto_gradcam_timer.isActive():

            interval_ms = self.get_gradcam_interval_ms()

            if interval_ms > 0:

                self.auto_gradcam_timer.start(interval_ms)


    def start_gradcam_analysis(self, trigger, source):

        if self.current_face_crop is None or self.current_frame is None:
            return


        face_snapshot = self.current_face_crop.copy()

        frame_snapshot = self.current_frame.copy()

        self.gradcam_worker = GradCAMWorker(
            face_crop_bgr=face_snapshot,
            annotated_frame_bgr=frame_snapshot,
            models_dict=self.models_dict,
            trigger=trigger,
            source=source,
        )

        self.gradcam_worker.analysis_ready.connect(self.on_gradcam_ready)

        self.gradcam_worker.error_occurred.connect(self.on_gradcam_error)

        self.gradcam_status_label.setText("Grad-CAM sedang diproses...")

        self.status_label.setText("Memproses Grad-CAM tiga model di background...")

        self.gradcam_worker.start(QThread.Priority.LowPriority)


    def on_gradcam_ready(self, analysis):
        self.latest_analysis = analysis
        if analysis["trigger"] == "auto":
            trigger_label = "Otomatis"
        elif analysis["trigger"] == "photo":
            trigger_label = "Foto"
        else:
            trigger_label = "Manual"
        self.gradcam_status_label.setText(
            f"Update: {analysis['finished_time'].strftime('%H:%M:%S')} | {trigger_label}"
        )
        self.status_label.setText("Proses Visualisasi Grad-CAM selesai.")
        for model_name in MODEL_PATHS_UI.keys():
            overlay = analysis["overlays"][model_name]
            pred = analysis["predictions"][model_name]
            self.gradcam_labels[model_name].set_bgr_image(overlay)
            self.gradcam_caption_labels[model_name].setText(
                f"{MODEL_LABELS_UI[model_name]}\n{pred['class'].upper()} ({pred['confidence'] * 100:.2f}%)"
            )

    def on_gradcam_error(self, message):

        self.gradcam_status_label.setText("Grad-CAM gagal.")

        self.status_label.setText(message)

        QMessageBox.critical(self, "Grad-CAM Error", message)


    def select_photo(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih Foto Wajah",
            str(PROJECT_ROOT),
            ("Image Files " "(*.png *.jpg *.jpeg *.bmp)"),
        )

        if not file_path:
            return


        self.current_source = "photo"

        self.stop_webcam()

        image_bgr = cv2.imread(file_path)

        if image_bgr is None:

            QMessageBox.warning(
                self,
                "Input Foto",
                "Foto tidak dapat dibaca."
            )

            return


        face_detector = load_face_detector()

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        display_image = image_bgr.copy()


        if len(faces) == 0:

            answer = QMessageBox.question(
                self,
                "Wajah Tidak Terdeteksi",
                (
                    "Wajah tidak terdeteksi.\n\n"
                    "Apakah seluruh gambar ingin "
                    "digunakan sebagai input?"
                ),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
            )

            if answer != QMessageBox.StandardButton.Yes:
                return

            face_crop = image_bgr.copy()

            predictions = utils.predict_face_crop_with_models(
                face_bgr=face_crop,
                models_dict=self.models_dict
            )


        else:

            largest_area = 0
            largest_box = None

            face_crop = None
            predictions = None


            for x, y, w, h in faces:

                current_face_crop = image_bgr[
                    y : y + h,
                    x : x + w
                ]

                if current_face_crop.size == 0:
                    continue


                current_predictions = (
                    utils.predict_face_crop_with_models(
                        face_bgr=current_face_crop,
                        models_dict=self.models_dict
                    )
                )


                face_area = w * h


                if face_area > largest_area:

                    largest_area = face_area

                    largest_box = (x, y, w, h)

                    face_crop = current_face_crop.copy()

                    predictions = current_predictions


                cv2.rectangle(
                    display_image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 255),
                    2
                )


                display_image = utils.draw_prediction_text(
                    frame=display_image,
                    x=x,
                    y=y,
                    predictions=current_predictions
                )


                if MODEL_3_KEY in current_predictions:

                    final_pred = current_predictions[
                        MODEL_3_KEY
                    ]

                    final_text = (
                        f"Final M3: "
                        f"{final_pred['class']} "
                        f"("
                        f"{final_pred['confidence']:.2f}"
                        f")"
                    )

                    cv2.putText(
                        display_image,
                        final_text,
                        (
                            x,
                            min(
                                display_image.shape[0] - 10,
                                y + h + 25
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )


            if (
                largest_box is None
                or face_crop is None
                or predictions is None
            ):

                QMessageBox.warning(
                    self,
                    "Input Foto",
                    "Wajah terdeteksi tetapi crop wajah tidak valid."
                )

                return


            x, y, w, h = largest_box

            cv2.rectangle(
                display_image,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                3
            )


        self.current_frame = display_image.copy()

        self.current_face_crop = face_crop.copy()

        self.current_predictions = predictions


        self.video_label.set_bgr_image(
            display_image
        )

        self.update_prediction_cards(
            predictions
        )


        if (
            self.gradcam_worker is not None
            and self.gradcam_worker.isRunning()
        ):

            self.gradcam_worker.wait()



        self.start_gradcam_analysis(
            trigger="photo",
            source="photo"
        )


    def save_latest_analysis(self):

        if self.latest_analysis is None:

            QMessageBox.warning(
                self, "Save", ("Belum ada hasil Grad-CAM " "yang dapat disimpan.")
            )

            return

        cfg.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

        analysis = self.latest_analysis

        timestamp = analysis["snapshot_time"].strftime("%Y%m%d_%H%M%S")

        prefix = "photo" if analysis.get("source") == "photo" else "webcam"


        if prefix == "webcam":

            paths = [
                cfg.SNAPSHOT_DIR / (f"webcam_snapshot_frame_" f"{timestamp}.png"),
                cfg.SNAPSHOT_DIR / (f"webcam_face_crop_original_" f"{timestamp}.png"),
                cfg.SNAPSHOT_DIR / (f"webcam_face_crop_grayscale_" f"{timestamp}.png"),
                cfg.SNAPSHOT_DIR / (f"webcam_face_input_224x224x3_" f"{timestamp}.png"),
                cfg.SNAPSHOT_DIR / (f"webcam_gradcam_comparison_" f"{timestamp}.png"),
            ]


        else:

            paths = [
                cfg.SNAPSHOT_DIR / (f"photo_snapshot_frame_" f"{timestamp}.png"),
                cfg.SNAPSHOT_DIR / (f"photo_face_crop_original_" f"{timestamp}.png"),
                cfg.SNAPSHOT_DIR / (f"photo_face_crop_grayscale_" f"{timestamp}.png"),
                cfg.SNAPSHOT_DIR / (f"photo_face_input_224x224x3_" f"{timestamp}.png"),
                cfg.SNAPSHOT_DIR / (f"photo_gradcam_comparison_" f"{timestamp}.png"),
            ]

        save_items = [
            analysis["annotated_frame_bgr"],
            analysis["face_crop_bgr"],
            analysis["face_gray_224"],
            analysis["input_bgr"],
            analysis["gradcam_comparison_bgr"],
        ]

        for path, image in zip(paths, save_items):

            ok = cv2.imwrite(str(path), image)

            if not ok:

                raise RuntimeError(f"Gagal menyimpan file: {path}")

        saved_names = "\n".join(path.name for path in paths)

        QMessageBox.information(
            self,
            "Save Berhasil",
            (
                f"5 file hasil klasifikasi berhasil disimpan ke:\n"
                f"{cfg.SNAPSHOT_DIR}\n\n"
                f"{saved_names}"
            ),
        )

        self.status_label.setText("5 file berhasil disimpan.")


    def on_worker_error(self, message):

        self.status_label.setText(message)

        QMessageBox.critical(self, "Error", message)


    def closeEvent(self, event: QCloseEvent):

        self.auto_gradcam_timer.stop()

        if self.webcam_worker is not None and self.webcam_worker.isRunning():

            self.webcam_worker.requestInterruption()

            self.webcam_worker.wait()

        if self.gradcam_worker is not None and self.gradcam_worker.isRunning():

            self.gradcam_worker.wait()

        event.accept()


# ============================================================
# MAIN
# ============================================================
def main():

    app = QApplication(sys.argv)

    window = FERDesktopApp()

    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()