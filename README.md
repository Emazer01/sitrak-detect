# Sitrak Shooting Target Detection Service

High-performance Computer Vision & Automated Scoring Backend Service built with **FastAPI**, **OpenCV**, and **NumPy**.

Layanan ini dirancang untuk menerima citra sasaran tembak (paper shooting target) dari front-end (web, mobile, atau streaming kamera), memproses koreksi perspektif citra, mendeteksi lingkaran target (bullseye & ring 1-10), mengidentifikasi lubang tembakan peluru, mengkalkulasi skor matematis, serta menghasilkan metrik analisis dispersi (grouping diameter) dan bias arah tembakan.

---

## Fitur Utama

- **Perspective Warp & Homography**: Otomatis mendeteksi batas kertas sasaran dan mengoreksi sudut pengambilan gambar yang miring/terdistorsi menjadi tampilan tegak (*canonical top-down view* $800 \times 800$ px).
- **Target & Bullseye Detection**: Deteksi titik pusat target (*center coordinates*) dan radius cincin konsentris scoring (Ring 1 hingga 10 / X-Ring) berbasis standar ISSF.
- **Bullet Hole Segmentation**: Deteksi akurat lubang tembakan peluru menggunakan kombinasi Adaptive Thresholding, Gradient Magnitude, Contour Geometry filtering (circularity, area, aspect ratio), serta Spatial Non-Maximum Suppression (NMS) untuk mencegah duplikasi.
- **Scoring Engine**:
  - Perhitungan jarak Euclidean dari pusat sasaran ke setiap lubang tembakan.
  - Skoring integer standar (10, 9, 8, dst.) dengan aturan inward-gauging (peluru menyentuh garis ring terhitung poin lebih tinggi).
  - Skoring desimal 10th-ring (misal 10.4, 9.8).
- **Analytics & Dispersion Generator**:
  - Ukuran pengelompokan (*grouping diameter / extreme spread*) dalam milimeter (mm).
  - *Mean Point of Impact* (MPI) dan bias arah tembakan (misal: `"top-left"`, `"top-right"`, `"bottom"`, `"centered"`).
  - Akurasi persentase terhadap skor maksimum.
- **Visual Overlay Generator**: Menggambar anotasi visual langsung pada gambar (titik tembak, nomor urut tembakan, badge skor per tembakan, lingkaran sebaran grouping, dan HUD summary) dan menyimpannya ke direktori publik `/static/annotated/`.
- **Dua Mode Deteksi**:
  1. `POST /api/v1/detect/upload` (Multipart image file upload untuk foto single shot/target).
  2. `POST /api/v1/detect/frame` (Base64 payload untuk stream frame kamera front-end berkala dengan deteksi delta tembakan baru).

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
│   │           ├── detect.py      # Endpoints /upload & /frame
│   │           └── health.py      # Endpoint /healthcheck
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic Settings & env configuration
│   │   └── exceptions.py          # Custom domain exceptions & global HTTP handler
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── detection.py           # Pydantic models (request & response JSON)
│   │   └── health.py              # Health check schema
│   ├── services/
│   │   ├── __init__.py
│   │   ├── scoring_engine.py      # Kalkulasi Euclidean, grouping mm, & directional bias
│   │   └── vision_service.py      # Pipeline OpenCV (warp, rings, holes, overlay)
│   └── storage/
│       ├── __init__.py
│       └── file_storage.py        # Pengelolaan penyimpanan file input & gambar anotasi
├── static/
│   ├── uploads/                   # Penyimpanan file foto target asli
│   └── annotated/                 # Penyimpanan file hasil anotasi Computer Vision
├── tests/
│   ├── __init__.py
│   └── test_vision.py             # Unit test dengan synthetic target generator
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
Isi konfigurasi pada file `.env` sesuai kebutuhan (misal setting CORS origin untuk frontend Anda).

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

### 1. Health Check
Memeriksa status operasional service.

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/healthcheck"
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "Sitrak Shooting Target Detection Service",
  "version": "1.0.0",
  "timestamp": "2026-10-08T03:15:20.123456Z",
  "gpu_available": false
}
```

---

### 2. Upload Target Image (`POST /api/v1/detect/upload`)
Mengunggah file foto sasaran tembak (JPEG/PNG).

**Request via cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect/upload" \
  -H "Accept: application/json" \
  -F "file=@/path/to/target_target.jpg" \
  -F "target_type=issf_standard_10_ring" \
  -F "caliber_mm=4.5" \
  -F "shooter_name=John Doe"
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
      "distance_from_center": 3.82,
      "decimal_score": 10.7
    },
    {
      "id": 2,
      "x": 372.0,
      "y": 380.4,
      "score": 9,
      "distance_from_center": 10.25,
      "decimal_score": 9.4
    }
  ],
  "analytics": {
    "grouping_diameter_mm": 35.4,
    "directional_bias": "top-left",
    "mean_point_of_impact_x": -6.2,
    "mean_point_of_impact_y": -4.8,
    "dispersion_radius_px": 50.6
  },
  "annotated_image_url": "/static/annotated/shot_result_abc123xyz.jpg",
  "metadata": {
    "perspective_corrected": true,
    "target_type": "issf_standard_10_ring",
    "caliber_mm": 4.5,
    "original_filename": "target_target.jpg",
    "shooter_name": "John Doe"
  }
}
```

---

### 3. Incremental Frame Detection (`POST /api/v1/detect/frame`)
Digunakan oleh front-end kamera/webcam untuk streaming frame secara periodik (base64) dan mendeteksi delta tembakan baru.

**Request via cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect/frame" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "target_type": "issf_standard_10_ring",
    "caliber_mm": 4.5,
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
    "grouping_diameter_mm": 21.8,
    "directional_bias": "centered"
  },
  "annotated_image_url": "/static/annotated/shot_result_delta_xyz.jpg",
  "metadata": {
    "frame_mode": "incremental_stream",
    "known_shots_count": 5,
    "new_shots_detected": 1
  }
}
```

---

## Integrasi Front-End

### 1. Rendering Gambar Hasil Anotasi
URL yang dikembalikan pada `annotated_image_url` dapat langsung digabungkan dengan base URL backend:
```html
<img src="http://localhost:8000/static/annotated/shot_result_abc123xyz.jpg" alt="Hasil Anotasi Tembakan" />
```

### 2. Konfigurasi CORS
Origin front-end umum (`localhost:3000`, `localhost:5173`) sudah diizinkan secara default. Untuk mengubah daftar origin di lingkungan staging/produksi, atur variabel `CORS_ORIGINS` di file `.env`:
```env
CORS_ORIGINS=["https://my-shooting-app.com","https://admin.my-shooting-app.com"]
```

---

## Menjalankan Unit & Integration Test
Proyek dilengkapi dengan test suite otomatis yang membuat target tembak sintetis dan menguji akurasi pipeline Computer Vision serta API endpoints:
```bash
pytest -v
```

