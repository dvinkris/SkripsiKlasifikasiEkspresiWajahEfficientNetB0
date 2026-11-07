
# 📖 Klasifikasi Ekspresi Wajah Menggunakan EfficientNetB0 pada Dataset FERPlus

Repository ini berisi implementasi penelitian skripsi mengenai klasifikasi ekspresi wajah menggunakan arsitektur **EfficientNetB0** dengan pendekatan *transfer learning* pada dataset **FERPlus**.

Penelitian dilakukan melalui tiga skenario pelatihan untuk menganalisis pengaruh *data augmentation* dan *fine-tuning* terhadap kinerja model. Evaluasi dilakukan menggunakan metrik klasifikasi, *confusion matrix*, dan visualisasi *Grad-CAM*. Penelitian ini juga dilengkapi aplikasi desktop berbasis PySide6 untuk mendemonstrasikan hasil klasifikasi ekspresi wajah.

## 📂 Dataset

Dataset yang digunakan adalah **FERPlus**, yang diperoleh dari Kaggle:

🔗 [Dataset FERPlus di Kaggle](https://www.kaggle.com/datasets/arnabkumarroy02/ferplus)

Dataset terdiri atas delapan kelas ekspresi wajah:

1. `angry`
2. `contempt`
3. `disgust`
4. `fear`
5. `happy`
6. `neutral`
7. `sad`
8. `surprise`

Citra sumber berukuran **112 × 112 piksel** dalam format *grayscale* dan diproses menjadi input RGB berukuran **224 × 224 × 3** untuk EfficientNetB0.

Struktur dataset yang digunakan:

```text
dataset/
├── training_full/
├── training_non_augmented/
├── validation_full/
├── validation_non_augmented/
└── test/
```

Setiap folder tersebut memiliki delapan subfolder berdasarkan nama kelas ekspresi wajah.

**Catatan:** Dataset tidak disertakan dalam repository karena ukurannya relatif besar. Pengguna perlu mengunduh dataset melalui tautan di atas dan menempatkannya sesuai struktur yang digunakan dalam penelitian.

## 🧪 Skenario Penelitian

Penelitian membandingkan tiga skenario pelatihan EfficientNetB0:

| Model | Data Pelatihan | Augmentasi Tambahan | Fine-Tuning |
|---|---|---|---|
| Model 1 | `training_non_augmented` | Tidak | Tidak |
| Model 2 | `training_full` | Ya | Tidak |
| Model 3 | `training_full` | Ya | Ya, 50 layer terakhir |

**Model 1 — Baseline Tanpa Augmentasi**

Menggunakan data `training_non_augmented`, tanpa augmentasi tambahan dan tanpa *fine-tuning*. Model ini digunakan sebagai pembanding awal.

**Model 2 — Dengan Data Augmentation**

Menggunakan data `training_full`, yang mencakup citra asli dan citra hasil augmentasi bawaan dataset, serta menerapkan augmentasi tambahan secara *online* selama pelatihan. Model ini tidak menggunakan *fine-tuning*.

**Model 3 — Dengan Data Augmentation dan Fine-Tuning**

Menggunakan data dan augmentasi tambahan yang sama dengan Model 2. Setelah tahap pelatihan awal, model dilanjutkan dengan *fine-tuning* pada 50 layer terakhir EfficientNetB0, dengan maksimum 30 epoch pada tahap tersebut.

Ketiga model menggunakan `validation_non_augmented` sebagai data validasi dan `test` yang sama untuk evaluasi akhir.

Arsitektur klasifikasi menggunakan **EfficientNetB0 dengan bobot awal ImageNet**, diikuti *Global Average Pooling*, *Batch Normalization*, *Dropout* sebesar 0,3, dan *Dense Softmax* untuk delapan kelas ekspresi.

## 📊 Hasil Evaluasi

Hasil *accuracy* ketiga model pada data uji FERPlus adalah:

| Model | Test Accuracy |
|---|---:|
| Model 1 — Tanpa Augmentasi | 66,13% |
| Model 2 — Dengan Augmentasi | 64,04% |
| **Model 3 — Augmentasi dan Fine-Tuning** | **78,84%** |

Berdasarkan hasil tersebut, **Model 3 memperoleh accuracy tertinggi**, yaitu **78,84%**.

Evaluasi penelitian juga mencakup:

- *Accuracy*
- *Precision*
- *Recall*
- *F1-score*
- *Confusion matrix*
- Visualisasi *Grad-CAM*

Metrik *precision*, *recall*, dan *F1-score* dianalisis untuk memberikan gambaran kinerja klasifikasi yang lebih lengkap. Visualisasi *Grad-CAM* digunakan untuk membantu menginterpretasikan area citra yang berkontribusi terhadap prediksi model.

## 🗂️ Struktur Repository

Repository menggunakan sembilan file coding utama yang disusun berdasarkan tahapan penelitian.

```text
skripsi-klasifikasi-ekspresi-wajah-efficientnetb0/
│
├── 01_cek_dataset.ipynb
├── 02_training_model_1_no_augmentation.ipynb
├── 03_training_model_2_with_augmentation.ipynb
├── 04_training_model_3_augmentation_finetuning.ipynb
├── 05_evaluasi_perbandingan_model.ipynb
├── 06_gradcam_perbandingan_model.ipynb
├── 07_demo_desktop.py
├── 08_utils.py
├── 09_config.py
│
├── outputs/
│   ├── models/
│   ├── figures/
│   └── results/
│
├── .gitignore
├── requirements.txt
└── README.md
```

Folder `outputs/` pada struktur tersebut menggambarkan bagian hasil yang direncanakan untuk disertakan dalam repository. Hasil eksperimen lainnya tetap disimpan secara lokal.

Fungsi masing-masing file coding:

| File | Fungsi |
|---|---|
| `01_cek_dataset.ipynb` | Pemeriksaan struktur, kualitas, distribusi, dan duplikasi dataset |
| `02_training_model_1_no_augmentation.ipynb` | Pelatihan Model 1 tanpa augmentasi tambahan |
| `03_training_model_2_with_augmentation.ipynb` | Pelatihan Model 2 dengan augmentasi |
| `04_training_model_3_augmentation_finetuning.ipynb` | Pelatihan dan *fine-tuning* Model 3 |
| `05_evaluasi_perbandingan_model.ipynb` | Evaluasi serta perbandingan ketiga model |
| `06_gradcam_perbandingan_model.ipynb` | Visualisasi dan analisis *Grad-CAM* |
| `07_demo_desktop.py` | Aplikasi desktop klasifikasi ekspresi wajah |
| `08_utils.py` | Kumpulan fungsi pendukung penelitian |
| `09_config.py` | Konfigurasi dataset, model, pelatihan, dan direktori output |

## ⚙️ Persiapan Lingkungan

Penelitian dikembangkan menggunakan **Python 3.10** dengan beberapa library utama, antara lain:

- TensorFlow dan Keras
- NumPy dan Pandas
- Matplotlib
- scikit-learn
- OpenCV
- PySide6
- Pillow

Versi library yang digunakan dicantumkan dalam file `requirements.txt`.

Untuk menginstal dependensi, jalankan perintah berikut pada terminal di direktori utama proyek:

```bash
python -m pip install -r requirements.txt
```

Notebook dapat dijalankan melalui Jupyter Notebook atau VS Code dengan ekstensi Jupyter dan kernel Python yang sesuai.

Pastikan seluruh dataset telah ditempatkan dalam folder `dataset/` sesuai konfigurasi pada `09_config.py`.

## 🚀 Menjalankan Penelitian

Jalankan notebook secara berurutan sesuai tahapan penelitian:

1. **Pemeriksaan Dataset**  
   `01_cek_dataset.ipynb`

2. **Pelatihan Model 1**  
   `02_training_model_1_no_augmentation.ipynb`

3. **Pelatihan Model 2**  
   `03_training_model_2_with_augmentation.ipynb`

4. **Pelatihan dan Fine-Tuning Model 3**  
   `04_training_model_3_augmentation_finetuning.ipynb`

5. **Evaluasi dan Perbandingan Model**  
   `05_evaluasi_perbandingan_model.ipynb`

6. **Visualisasi Grad-CAM**  
   `06_gradcam_perbandingan_model.ipynb`

Seluruh notebook menggunakan konfigurasi dan fungsi pendukung yang didefinisikan dalam `09_config.py` dan `08_utils.py`.

**Catatan:** Pelatihan ulang memerlukan dataset dan sumber daya komputasi yang memadai. Notebook evaluasi dan *Grad-CAM* dapat menggunakan checkpoint model terbaik yang telah dihasilkan sebelumnya. Beberapa analisis *Grad-CAM* juga memerlukan hasil prediksi dari tahap evaluasi.

## 💾 Checkpoint Model

Checkpoint terbaik dari masing-masing model disimpan pada folder `outputs/models/` dengan nama:

```text
outputs/models/
├── model_1_baseline_no_augmentation_best.keras
├── model_2_training_full_with_augmentation_best.keras
└── model_3_training_full_augmentation_finetuning_best.keras
```

Ketiga checkpoint tersebut digunakan untuk evaluasi, visualisasi *Grad-CAM*, serta aplikasi desktop.

Path checkpoint didefinisikan dalam file `09_config.py`.

## 🖥️ Aplikasi Desktop

Penelitian ini dilengkapi aplikasi desktop berbasis **PySide6** untuk mendemonstrasikan klasifikasi ekspresi wajah menggunakan model yang telah dilatih.

Fitur aplikasi meliputi:

- Klasifikasi ekspresi wajah melalui **webcam**.
- Klasifikasi ekspresi wajah melalui **input foto**.
- Penampilan hasil prediksi dari ketiga model.
- Penggunaan **Model 3** sebagai hasil klasifikasi utama.
- Visualisasi **Grad-CAM** untuk membandingkan perhatian ketiga model.
- Pemrosesan beberapa wajah pada input foto, dengan wajah terbesar sebagai fokus utama tampilan analisis.

Untuk menjalankan aplikasi, buka terminal pada direktori utama proyek dan jalankan:

```bash
python 07_demo_desktop.py
```

**Catatan:** Pastikan ketiga checkpoint model terbaik telah tersedia pada direktori `outputs/models/` dan dependensi Python sudah terinstal.

## 📚 Referensi Dataset

Barsoum, E., Zhang, C., Canton Ferrer, C., & Zhang, Z. (2016). *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution*. ACM International Conference on Multimodal Interaction.

- [FERPlus — Microsoft GitHub](https://github.com/microsoft/FERPlus)
- [FERPlus Dataset — Kaggle](https://www.kaggle.com/datasets/arnabkumarroy02/ferplus)

---

**Repository ini disusun sebagai dokumentasi implementasi dan hasil eksperimen penelitian skripsi klasifikasi ekspresi wajah menggunakan EfficientNetB0 pada dataset FERPlus.**
