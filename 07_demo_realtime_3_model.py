import sys
import importlib.util
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf


def find_project_root(start_path=None):
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)

    current = start_path.resolve()

    while True:
        if (current / "09_config.py").exists() and (current / "08_utils.py").exists():
            return current

        if current.parent == current:
            raise FileNotFoundError(
                "Project root tidak ditemukan. Pastikan 09_config.py dan 08_utils.py tersedia."
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
cfg = import_py_file("ferplus_config", PROJECT_ROOT / "09_config.py")
utils = import_py_file("ferplus_utils", PROJECT_ROOT / "08_utils.py")

utils.set_global_seed(cfg.SEED)
utils.ensure_directories()


def load_demo_models():
    model_map = {
        "M1 Baseline": cfg.MODEL_1_BEST_PATH,
        "M2 Aug": cfg.MODEL_2_BEST_PATH,
        "M3 Aug+FT": cfg.MODEL_3_BEST_PATH
    }

    models_dict = {}

    for model_name, model_path in model_map.items():
        model_path = Path(model_path)

        print(f"Loading {model_name}: {model_path}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan: {model_path}\n"
                f"Pastikan training sudah selesai dan model best sudah tersimpan."
            )

        models_dict[model_name] = tf.keras.models.load_model(model_path)

        print(f"{model_name} loaded.")

    return models_dict


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise RuntimeError("Haar Cascade face detector gagal dimuat.")

    return face_detector


def add_title_to_image(image_bgr, title_lines):
    image_bgr = cv2.resize(image_bgr, cfg.IMAGE_SIZE)

    title_height = 75
    title_area = 255 * np.ones((title_height, image_bgr.shape[1], 3), dtype=np.uint8)

    y = 22

    for line in title_lines:
        cv2.putText(
            title_area,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        y += 22

    return cv2.vconcat([title_area, image_bgr])


def create_demo_ui_frame(display_frame, predictions=None, face_detected=False, status_text=""):
    display_frame = cv2.resize(display_frame, (960, 720))
    
    frame_height, frame_width = display_frame.shape[:2]
    panel_width = 420

    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (35, 35, 35)

    y = 35

    cv2.putText(
        panel,
        "FERPlus Expression Demo",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    y += 35

    cv2.putText(
        panel,
        "EfficientNetB0 - 3 Model Comparison",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )

    y += 45

    face_status = "Detected" if face_detected else "Not detected"
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)

    cv2.putText(
        panel,
        f"Face: {face_status}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        face_color,
        2,
        cv2.LINE_AA
    )

    y += 45

    cv2.putText(
        panel,
        "Prediction:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    y += 35

    if predictions is not None:
        for model_name, pred in predictions.items():
            pred_text = f"{model_name}: {pred['class']} ({pred['confidence']:.2f})"

            color = (255, 255, 255)

            if model_name == "M3 Aug+FT":
                color = (0, 255, 255)

            cv2.putText(
                panel,
                pred_text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA
            )

            y += 28

        if "M3 Aug+FT" in predictions:
            y += 15

            final_pred = predictions["M3 Aug+FT"]

            cv2.putText(
                panel,
                "Final Result:",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            y += 35

            cv2.putText(
                panel,
                f"{final_pred['class']} ({final_pred['confidence']:.2f})",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            y += 45
    else:
        cv2.putText(
            panel,
            "No prediction yet",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        y += 40

    y += 20

    cv2.line(panel, (20, y), (panel_width - 20, y), (100, 100, 100), 1)

    y += 35

    cv2.putText(
        panel,
        "Controls:",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    y += 35

    cv2.putText(
        panel,
        "q : Quit demo",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    y += 28

    cv2.putText(
        panel,
        "s : Save snapshot + Grad-CAM",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    y += 45

    if status_text:
        cv2.putText(
            panel,
            status_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    ui_frame = cv2.hconcat([display_frame, panel])

    return ui_frame


def save_snapshot_with_gradcam(display_frame, face_crop, models_dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    frame_path = cfg.SNAPSHOT_DIR / f"webcam_snapshot_frame_{timestamp}.png"
    face_crop_path = cfg.SNAPSHOT_DIR / f"webcam_face_crop_original_{timestamp}.png"
    face_gray_path = cfg.SNAPSHOT_DIR / f"webcam_face_crop_grayscale_{timestamp}.png"
    face_input_path = cfg.SNAPSHOT_DIR / f"webcam_face_input_224x224x3_{timestamp}.png"
    gradcam_path = cfg.SNAPSHOT_DIR / f"webcam_gradcam_comparison_{timestamp}.png"

    # 1. Simpan frame webcam yang sudah ada bounding box dan teks prediksi
    cv2.imwrite(str(frame_path), display_frame)

    # 2. Simpan crop wajah asli dari webcam
    cv2.imwrite(str(face_crop_path), face_crop)

    # 3. Simpan crop wajah grayscale
    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    face_gray_224 = cv2.resize(face_gray, cfg.IMAGE_SIZE)
    cv2.imwrite(str(face_gray_path), face_gray_224)

    # 4. Buat dan simpan input akhir ke model: grayscale -> RGB 3 channel -> 224x224x3
    input_arr = utils.preprocess_face_crop(face_crop)
    input_rgb = input_arr[0].astype("uint8")
    input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(face_input_path), input_bgr)

    # 5. Buat Grad-CAM gabungan
    panels = []

    original_panel = add_title_to_image(
        cv2.resize(face_crop, cfg.IMAGE_SIZE),
        ["Original", "Face crop"]
    )
    panels.append(original_panel)

    input_panel = add_title_to_image(
        input_bgr,
        ["Input model", "224x224x3 grayscale"]
    )
    panels.append(input_panel)

    for model_name, model in models_dict.items():
        heatmap, pred_index, probs, used_conv_layer = utils.make_gradcam_heatmap(
            img_array=input_arr,
            model=model,
            pred_index=None,
            last_conv_layer_name=None
        )

        pred_class = cfg.CLASS_NAMES[pred_index]
        confidence = float(probs[pred_index])

        overlay_bgr = utils.make_gradcam_overlay(
            image_path=face_input_path,
            heatmap=heatmap,
            alpha=0.4
        )

        panel = add_title_to_image(
            overlay_bgr,
            [
                model_name,
                f"{pred_class} ({confidence:.2f})",
                f"Layer: {used_conv_layer}"
            ]
        )

        panels.append(panel)

    gradcam_comparison = cv2.hconcat(panels)
    cv2.imwrite(str(gradcam_path), gradcam_comparison)

    print("Snapshot frame disimpan ke:", frame_path)
    print("Crop wajah asli disimpan ke:", face_crop_path)
    print("Crop wajah grayscale disimpan ke:", face_gray_path)
    print("Input model 224x224x3 disimpan ke:", face_input_path)
    print("Grad-CAM gabungan disimpan ke:", gradcam_path)


def main():
    print("Class order:")
    for idx, class_name in enumerate(cfg.CLASS_NAMES):
        print(idx, class_name)

    print("\nMemuat model...")
    models_dict = load_demo_models()

    print("\nMemuat face detector...")
    face_detector = load_face_detector()

    print("\nMembuka webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam tidak bisa dibuka. Cek kamera atau permission.")

    print("Demo berjalan.")
    print("Tekan 'q' untuk keluar.")
    print("Tekan 's' untuk menyimpan snapshot lengkap + Grad-CAM.")

    last_frame = None
    last_face_crop = None
    status_text = ""

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame webcam tidak terbaca.")
            break

        display_frame = frame.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60)
        )

        current_face_crop_for_snapshot = None
        current_predictions_for_ui = None
        current_largest_area = 0

        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]

            if face_crop.size == 0:
                continue

            face_area = w * h

            predictions = utils.predict_face_crop_with_models(
                face_bgr=face_crop,
                models_dict=models_dict
            )

            if face_area > current_largest_area:
                current_largest_area = face_area
                current_face_crop_for_snapshot = face_crop.copy()
                current_predictions_for_ui = predictions

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            display_frame = utils.draw_prediction_text(
                frame=display_frame,
                x=x,
                y=y,
                predictions=predictions
            )

            if "M3 Aug+FT" in predictions:
                final_pred = predictions["M3 Aug+FT"]
                final_text = f"Final M3: {final_pred['class']} ({final_pred['confidence']:.2f})"

                cv2.putText(
                    display_frame,
                    final_text,
                    (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        face_detected = current_face_crop_for_snapshot is not None

        if current_face_crop_for_snapshot is not None:
            last_face_crop = current_face_crop_for_snapshot.copy()
        else:
            last_face_crop = None

        last_frame = display_frame.copy()

        ui_frame = create_demo_ui_frame(
            display_frame=display_frame,
            predictions=current_predictions_for_ui,
            face_detected=face_detected,
            status_text=status_text
        )

        cv2.imshow("FERPlus EfficientNetB0 - 3 Model Demo", ui_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Keluar dari demo.")
            break

        if key == ord("s"):
            if last_frame is None:
                print("Tidak ada frame untuk disimpan.")
                status_text = "No frame to save."
                continue

            if last_face_crop is None:
                print("Tidak ada wajah terdeteksi untuk dibuat Grad-CAM.")
                status_text = "No face detected."
                continue

            save_snapshot_with_gradcam(
                display_frame=last_frame,
                face_crop=last_face_crop,
                models_dict=models_dict
            )

            status_text = "Snapshot + Grad-CAM saved."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()