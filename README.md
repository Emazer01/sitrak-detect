# Sitrak Shooting Target Detection Service

High-performance Computer Vision, Deep Learning (YOLO26 & YOLO11), and Automated Scoring Backend Service built with **FastAPI**, **Ultralytics YOLO**, **OpenCV**, and **NumPy**.

Layanan ini dirancang untuk menerima citra sasaran tembak dari front-end (aplikasi web, mobile, atau streaming kamera pengawas lesan), memproses koreksi dimensi dan perspektif citra, mendeteksi lingkaran target (bullseye & cincin skor 1-10), mengidentifikasi lubang/titik tembakan peluru (termasuk perkenaan yang berdekatan atau menempel), mengkalkulasi skor matematis standar ISSF, serta menghasilkan metrik analisis dispersi (*grouping diameter*) dan bias arah tembakan.

Mendukung sasaran tembak fisik (*paper shooting target* dengan lubang sobekan peluru) maupun sasaran tembak elektronik/monitor digital (seperti Krüger 50-Foot Pistol Target dengan titik putih/cyan pada bullseye hitam dan titik biru/abu-abu pada cincin skor).

---

## Fitur Utama

- **Auto-Crop 1:1 Aspect Ratio (Bujur Sangkar)**:
  Otomatis mendeteksi pusat lesan tembak dan memotong (*crop*) foto berorientasi *landscape* maupun *portrait* menjadi rasio aspek tepat 1:1 berpusat pada target. Mencegah distorsi bentuk lingkaran target sebelum diproses ke homografi.
- **Perspective Warp & Homography**:
  Otomatis mendeteksi batas kertas sasaran dan mengoreksi sudut pengambilan foto yang miring (*slanted/skewed*) menjadi tampilan tegak lurus sempurna (*canonical top-down view* $800 \times 800$ px).
- **Multi-Engine Detection Support**:
  Mendukung fleksibilitas pemilihan mesin deteksi melalui parameter API `engine`:
  1. `yolo11` (`models/bullet_hole_yolo11.pt`): Model Ultralytics YOLO11 akurasi tinggi (mencapai 100% pada perkenaan tunggal & ganda menempel).
  2. `yolo26` (`models/bullet_hole_yolo26.pt`): Model spesialis deteksi target kecil & pola digital Krüger.
  3. `model_sendiri` (`models/bullet_hole_yolo.pt`): Model default produksi berbasis AI YOLO yang dipadukan dengan ensemble filter Computer Vision.
  4. `cv`: Classical Computer Vision (DoG + MSER + Adaptive Gradient + Watershed).
- **Dekomposisi Perkenaan Menempel (*Adjacent / Touching Shots / Double-Taps*)**:
  - Mengantisipasi tembakan berhimpitan (*cloverleaf / double-taps*) menggunakan analisis kontur *Distance Transform* dan pencarian *local intensity peaks*.
  - Ambang batas penggabungan spasial (*merge threshold*) diperketat ke `4.5 px` agar dua tembakan yang menempel (jarak 7–12 px) **tidak dilebur menjadi satu**.
  - Penyesuaian ambang batas IoU NMS (`0.58`) agar kotak deteksi yang bersebelahan tidak saling mematikan.
- **Scoring Engine Standar ISSF**:
  - Perhitungan jarak Euclidean dari pusat sasaran ke setiap pusat tembakan peluru.
  - Skoring integer standar (10, 9, 8, dst.) dengan aturan resmi *inward-gauging* (peluru yang menyentuh batas garis ring dalam dihitung nilai poin yang lebih tinggi).
  - Skoring desimal 10th-ring (misalnya 10.4, 9.8).
- **Analytics & Dispersion Metrics**:
  - Ukuran pengelompokan tembakan (*grouping diameter / extreme spread*) dalam milimeter (mm).
  - *Mean Point of Impact* (MPI) dan bias arah tembakan (misal: `"top-left"`, `"top-right"`, `"centered"`).
  - Akurasi persentase terhadap skor maksimum.
- **Visual Overlay Generator**:
  Menggambar anotasi visual instan pada gambar (titik tembak, nomor urut tembakan, badge skor per tembakan, lingkaran sebaran grouping, dan HUD ringkasan performa) lalu menyimpannya ke direktori publik `/static/annotated/`.
- **Dua Mode Deteksi Endpoint**:
  1. `POST /api/v1/detect` (atau `/upload`): Multipart form-data untuk upload foto sasaran (JPEG/PNG/WebP).
  2. `POST /api/v1/detect/frame`: Payload JSON Base64 untuk stream frame kamera webcam/video secara berkala dengan pelacakan delta tembakan baru.

---

## Struktur Direktori

```text
sitrak-detect/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Entrypoint FastAPI, CORS middleware, mounting static
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py              # Router aggregator
│   │   └── v1/
│   │       ├── __init__.py
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── detect.py      # Endpoints /detect, /frame, /engines, /model-status, /model-reload
│   │           └── health.py      # Endpoint /healthcheck
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings & konfigurasi .env
│   │   └── exceptions.py          # Custom domain exceptions & global HTTP handler
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── detection.py           # Schema request & response JSON
│   │   └── health.py              # Schema health check
│   ├── services/
│   │   ├── __init__.py
│   │   ├── scoring_engine.py      # Kalkulasi Euclidean, grouping mm, & directional bias
│   │   └── vision_service.py      # Core engine Computer Vision & YOLO inference
│   └── storage/
│       ├── __init__.py
│       └── file_storage.py        # Penyimpanan file upload mentah & hasil anotasi
├── models/
│   ├── bullet_hole_yolo.pt        # Model default produksi
│   ├── bullet_hole_yolo26.pt      # Bobot terlatih arsitektur YOLO26
│   └── bullet_hole_yolo11.pt      # Bobot terlatih arsitektur YOLO11
├── static/
│   ├── uploads/                   # Foto sasaran asli yang diunggah
│   └── annotated/                 # Hasil gambar yang telah dianotasi visual
├── tests/
│   ├── __init__.py
│   └── test_vision.py             # 14 unit test lengkap (endpoints, model, auto-crop, skoring)
├── training/
│   ├── build_complete_target_dataset.py # Generator dataset lengkap (augmentasi + touching dots)
│   ├── train_final_models.py      # Script pelatihan YOLO26 dan YOLO11
│   ├── evaluate_12_user_photos.py # Script evaluasi end-to-end 12 foto sasaran tembak
│   ├── evaluate_models.py         # Script benchmark perbandingan model
│   └── assets/                    # Master template lesan Krüger & ground-truth user
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Instalasi & Menjalankan Server

### 1. Prasyarat
- Python 3.10 atau lebih baru.
- Virtual Environment tool (`venv` atau `conda`).

### 2. Setup Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Konfigurasi Environment
Salin file `.env.example` menjadi `.env`:
```bash
cp .env.example .env
```

Sesuaikan nilai pada file `.env` bila diperlukan:
```env
# Server & CORS
HOST="0.0.0.0"
PORT=8000
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173","http://127.0.0.1:3000","http://127.0.0.1:5173"]

# Detection Engine Default ("hybrid", "yolo11", "yolo26", "model_sendiri", "ai", "cv")
DETECTION_ENGINE="hybrid"
AI_CONFIDENCE_THRESHOLD=0.15
AI_IOU_THRESHOLD=0.58
```

> **Catatan Keamanan Frontend:** URL API backend tidak boleh disimpan di `localStorage` peramban. Frontend harus membaca URL API dari file `.env` front-end (`VITE_API_BASE_URL` atau `NEXT_PUBLIC_API_URL`).

### 5. Menjalankan Server
Jalankan service menggunakan Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Service akan aktif di:
- **API Base URL**: `http://localhost:8000`
- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## Dokumentasi API & Contoh Request

### 1. Daftar Mesin Deteksi (`GET /api/v1/detect/engines`)
Melihat daftar mesin deteksi yang terpasang dan status ketersediaannya.

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/detect/engines"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "default_engine": "hybrid",
  "engines": [
    {
      "id": "model_sendiri",
      "name": "Model Sendiri (Default AI Ensemble)",
      "type": "local_deep_learning",
      "model": "bullet_hole_yolo.pt",
      "is_available": true
    },
    {
      "id": "yolo26",
      "name": "YOLO26 (Small-Target Puncture Detector)",
      "type": "local_deep_learning",
      "model": "bullet_hole_yolo26.pt",
      "is_available": true
    },
    {
      "id": "yolo11",
      "name": "YOLO11 (Ultralytics High-Precision Detector)",
      "type": "local_deep_learning",
      "model": "bullet_hole_yolo11.pt",
      "is_available": true
    }
  ]
}
```

---

### 2. Upload Target Image (`POST /api/v1/detect`)
Mengunggah file foto sasaran tembak untuk dianalisis dan dinilai.

**Request via cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Accept: application/json" \
  -F "file=@/path/to/target_photo.jpg" \
  -F "engine=yolo11" \
  -F "caliber_mm=4.5" \
  -F "target_type=issf_standard_10_ring" \
  -F "shooter_name=Penembak 1"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "target_detected": true,
  "total_score": 87,
  "max_possible_score": 100,
  "accuracy_percentage": 87.0,
  "shots_count": 10,
  "shots": [
    {
      "id": 1,
      "x": 395.2,
      "y": 392.1,
      "score": 10,
      "decimal_score": 10.7,
      "distance_from_center": 3.82,
      "clock_direction": "11:00",
      "confidence": 0.88,
      "detection_source": "yolo11_ai"
    },
    {
      "id": 2,
      "x": 372.0,
      "y": 380.4,
      "score": 9,
      "decimal_score": 9.4,
      "distance_from_center": 10.25,
      "clock_direction": "10:30",
      "confidence": 0.82,
      "detection_source": "yolo11_ai_double_tap"
    }
  ],
  "analytics": {
    "grouping_diameter_mm": 28.4,
    "directional_bias": "top-left",
    "mean_point_of_impact_x": -3.2,
    "mean_point_of_impact_y": -4.1,
    "dispersion_radius_px": 40.5
  },
  "annotated_image_url": "/static/annotated/annotated_abc123.jpg",
  "detection_engine_used": "yolo11",
  "metadata": {
    "perspective_corrected": true,
    "cropped_to_square": true,
    "crop_box": {
      "x": 0,
      "y": 200,
      "width": 1200,
      "height": 1200,
      "original_width": 1200,
      "original_height": 1600
    },
    "target_type": "issf_standard_10_ring",
    "caliber_mm": 4.5,
    "scale_mm_per_pixel": 0.35,
    "selected_engine": "yolo11"
  }
}
```

---

### 3. Incremental Frame Detection (`POST /api/v1/detect/frame`)
Digunakan untuk pemrosesan berkala dari frame kamera streaming / webcam front-end dalam format base64.

**Request via cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect/frame" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "engine": "model_sendiri",
    "caliber_mm": 4.5,
    "target_type": "issf_standard_10_ring",
    "known_shots_count": 5
  }'
```

**Response (200 OK):**
```json
{
  "status": "success",
  "target_detected": true,
  "total_score": 58,
  "max_possible_score": 60,
  "accuracy_percentage": 96.7,
  "shots_count": 6,
  "shots": [...],
  "analytics": {
    "grouping_diameter_mm": 19.5,
    "directional_bias": "centered"
  },
  "annotated_image_url": "/static/annotated/shot_result_xyz.jpg",
  "detection_engine_used": "model_sendiri",
  "metadata": {
    "frame_mode": "incremental_stream",
    "known_shots_count": 5,
    "new_shots_detected": 1
  }
}
```

---

### 4. Status & Hot-Reload Model
- **Cek Status Model Aktif**:
  ```bash
  curl -X GET "http://localhost:8000/api/v1/detect/model-status"
  ```
- **Hot-Reload Bobot Model Baru (Tanpa Restart Server)**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/detect/model-reload"
  ```

---

## Menjalankan Unit & Integration Test

Seluruh rangkaian pengujian sistem mencakup validasi endpoint, integrasi YOLO26 & YOLO11, auto-crop 1:1, dan engine matematika skoring:
```bash
pytest tests/test_vision.py -v
```

Hasil uji: **14/14 tests PASSED (100%)**.

---

## Pipeline Pelatihan & Evaluasi Model

### 1. Membangun Dataset Pelatihan Final
Generator membuat dataset gabungan yang mengikutsertakan sampel ground-truth pengguna, augmentasi multi-sudut, pola tembakan berhimpitan (*touching pairs & triplets*), titik samar abu-abu, dan *hard negatives* untuk pencegahan *false positive* pada angka sasaran:
```bash
python training/build_complete_target_dataset.py
```
Output dataset tersimpan di `training/dataset_final/` (893 gambar train, 201 gambar val).

### 2. Melatih Model YOLO26 & YOLO11
Script pelatihan akan melakukan *transfer learning / fine-tuning* pada model YOLO26 dan YOLO11, kemudian otomatis menyalin bobot terbaik (*best.pt*) ke direktori `models/`:
```bash
python training/train_final_models.py --epochs 8 --batch 16 --workers 2
```
File bobot yang diperbarui:
- `models/bullet_hole_yolo26.pt`
- `models/bullet_hole_yolo11.pt`
- `models/bullet_hole_yolo.pt` (produksi default)

### 3. Evaluasi End-to-End pada 12 Foto Smartphone Pengguna
Menguji akurasi deteksi langsung pada 12 foto smartphone hasil jepretan layar monitor sasaran:
```bash
python training/evaluate_12_user_photos.py
```
Tingkat akurasi **Exact / Close ($\pm 1$)** mencapai **83.3%** pada YOLO11, dengan keberhasilan pemisahan tembakan menempel/berdekatan (*double-taps*).
