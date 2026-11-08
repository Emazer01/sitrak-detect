# Sitrak Shooting Target Detection Service

Backend Service berbasis **FastAPI**, **Ultralytics YOLO (YOLO11 & YOLO26)**, **OpenCV**, dan **NumPy** untuk deteksi otomatis perkenaan peluru, koreksi perspektif lesan, pengabaian stiker tambalan (*target pasters*), kalkulasi skor standar ISSF, dan analisis sebaran tembakan (*grouping diameter*).

Layanan ini beroperasi **100% lokal (*self-hosted / offline-ready*)** tanpa ketergantungan API eksternal pihak ketiga, dirancang untuk diintegrasikan dengan aplikasi web, mobile, maupun sistem monitoring kamera sasaran tembak *real-time*.

---

## Daftar Isi

1. [Fitur Utama](#fitur-utama)
2. [Arsitektur & Spesifikasi Model](#arsitektur--spesifikasi-model)
3. [Struktur Direktori](#struktur-direktori)
4. [Tata Cara Instalasi di Device Lain](#tata-cara-instalasi-di-device-lain)
   - [A. Prasyarat Sistem](#a-prasyarat-sistem)
   - [B. Kloning / Salin Proyek](#b-kloning--salin-proyek)
   - [C. Setup Virtual Environment](#c-setup-virtual-environment)
   - [D. Instalasi Dependensi Python](#d-instalasi-dependensi-python)
   - [E. Catatan Khusus Dependensi OS (Linux / Ubuntu)](#e-catatan-khusus-dependensi-os-linux--ubuntu)
   - [F. Akselerasi GPU NVIDIA (Opsional)](#f-akselerasi-gpu-nvidia-opsional)
   - [G. Verifikasi File Model Bobot](#g-verifikasi-file-model-bobot)
   - [H. Konfigurasi Environment (`.env`)](#h-konfigurasi-environment-env)
5. [Menjalankan Server](#menjalankan-server)
   - [Mode Development](#1-mode-development)
   - [Mode Production (Uvicorn Multi-Workers)](#2-mode-production-uvicorn-multi-workers)
   - [Mode Production Linux Service (Systemd)](#3-mode-production-linux-service-systemd)
6. [Akses Multi-Server & Remote Jaringan](#akses-multi-server--remote-jaringan)
   - [Akses Jaringan Lokal (LAN / Wi-Fi)](#1-akses-jaringan-lokal-lan--wi-fi)
   - [Akses Remote Server via SSH Tunnel](#2-akses-remote-server-via-ssh-tunnel)
   - [Integrasi Frontend & Keamanan](#3-integrasi-frontend--keamanan)
7. [Dokumentasi API & Contoh Request](#dokumentasi-api--contoh-request)
8. [Pengujian & Verifikasi (Unit Tests)](#pengujian--verifikasi-unit-tests)
9. [Pipeline Pelatihan Ulang Model](#pipeline-pelatihan-ulang-model)

---

## Fitur Utama

- **100% Lokal & Bebas Ketergantungan Eksternal**:
  Seluruh inferensi Deep Learning (YOLO11, YOLO26) dan Computer Vision berjalan langsung pada CPU/GPU server lokal tanpa membutuhkan koneksi internet atau API eksternal.
- **Pengabaian Stiker Tambalan Sasaran (*Paster / Tambalan Suppression*)**:
  Model dilatih dengan *hard-negative target repair pasters* (stiker lingkaran matte warna hitam di bullseye maupun krem di cincin luar). Sistem **100% mengabaikan tambalan** dan hanya mendeteksi lubang tembakan nyata.
- **Dukungan Ganda Sasaran (Fisik & Digital)**:
  - **Sasaran Fisik Kertas**: Mendeteksi sobekan serat kertas peluru nyata (*ragged polygon tears*) dengan kontras karton penahan di belakangnya.
  - **Sasaran Monitor Digital**: Mendeteksi titik tembakan elektronik (seperti Krüger 50-Foot Pistol Target dengan titik putih/cyan pada bullseye hitam dan titik biru/abu-abu pada cincin skor).
- **Auto-Crop 1:1 Aspect Ratio (Bujur Sangkar)**:
  Otomatis mendeteksi pusat lesan tembak dan memotong (*crop*) foto berorientasi *landscape* maupun *portrait* menjadi rasio tepat 1:1 berpusat pada target sebelum pemrosesan homografi, mencegah distorsi elips.
- **Perspective Warp & Homography Correction**:
  Otomatis mendeteksi batas kertas sasaran dan mengoreksi sudut pengambilan foto yang miring (*slanted/skewed*) menjadi tampilan tegak lurus sempurna (*canonical top-down view* $800 \times 800$ px).
- **Dekomposisi Tembakan Menempel (*Adjacent / Touching Shots / Double-Taps*)**:
  Menggunakan analisis kontur *Distance Transform* dan pencarian puncak intensitas lokal (*local peaks*) dengan ambang batas *merge* diperketat ke `4.5 px` agar dua lubang peluru yang menempel/berhimpitan **tidak dilebur menjadi satu**.
- **Scoring Engine Standar ISSF**:
  - Perhitungan jarak Euclidean dari pusat sasaran ke setiap pusat tembakan peluru.
  - Skoring integer standar (10, 9, 8, dst.) dengan aturan resmi *inward-gauging* (peluru yang menyentuh batas garis ring dalam memperoleh poin yang lebih tinggi).
  - Skoring desimal 10th-ring (misalnya 10.4, 9.8).
- **Analisis Statistik & Dispersi (*Grouping Diameter*)**:
  - Ukuran pengelompokan tembakan (*grouping diameter / extreme spread*) dalam satuan milimeter (mm).
  - *Mean Point of Impact* (MPI) dan bias arah sebaran tembakan (contoh: `"top-left"`, `"bottom-right"`, `"centered"`).
  - Persentase akurasi terhadap skor maksimum.
- **Visual Overlay Generator**:
  Menggambar anotasi visual instan (titik tembak, nomor urut tembakan, badge skor per lubang, lingkaran sebaran grouping, dan HUD ringkasan performa) lalu menyimpannya ke direktori publik `/static/annotated/`.

---

## Arsitektur & Spesifikasi Model

Sistem menyediakan 4 pilihan mesin deteksi yang dapat dipilih via parameter API `engine`:

| Engine ID | Model File | Deskripsi Karakteristik |
| :--- | :--- | :--- |
| **`model_sendiri`** *(Default)* | `models/bullet_hole_yolo.pt` | Model produksi teroptimasi hasil evaluasi terbaik, menggabungkan deteksi Deep Learning dan ensemble Computer Vision. |
| **`yolo11`** | `models/bullet_hole_yolo11.pt` | Arsitektur Ultralytics YOLO11 akurasi tinggi, sangat presisi membedakan sobekan kertas nyata dari stiker tambalan. |
| **`yolo26`** | `models/bullet_hole_yolo26.pt` | Arsitektur YOLO26 dengan alokasi target kecil, sensitif terhadap titik digital dan perkenaan tipis. |
| **`cv`** | *N/A (Algoritmik)* | Algoritma Computer Vision klasik tanpa AI (DoG + MSER + Adaptive Gradient + Watershed). |

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
│   ├── bullet_hole_yolo11.pt      # Bobot terlatih arsitektur YOLO11
│   └── bullet_hole_yolo26.pt      # Bobot terlatih arsitektur YOLO26
├── static/
│   ├── uploads/                   # Foto sasaran asli yang diunggah
│   └── annotated/                 # Hasil gambar yang telah dianotasi visual
├── tests/
│   ├── __init__.py
│   └── test_vision.py             # 14 unit test lengkap (endpoints, model, auto-crop, skoring)
├── training/
│   ├── build_complete_target_dataset.py # Generator dataset (augmentasi + paper tears + hard-negative tambalan)
│   ├── train_final_models.py      # Script pelatihan YOLO26 dan YOLO11
│   ├── evaluate_12_user_photos.py # Script evaluasi 12 foto sasaran tembak
│   └── assets/                    # Master template lesan Krüger & sampel fisik nyata
├── .env.example                   # Template konfigurasi environment
├── .gitignore                     # Filter file yang tidak perlu di-push ke git
├── requirements.txt               # Daftar paket dependensi Python
└── README.md
```

---

## Tata Cara Instalasi di Device Lain

Panduan ini berlaku untuk pemasangan pada laptop/PC pengembang, server lokal mini-PC (Intel NUC, Raspberry Pi 5), maupun server Cloud/VPS (Ubuntu/Debian).

### A. Prasyarat Sistem
- **Sistem Operasi**: Windows 10/11, Linux (Ubuntu 20.04/22.04/24.04, Debian 11/12), atau macOS.
- **Python**: Versi `3.10`, `3.11`, atau `3.12` (disarankan).
- **RAM**: Minimal 4 GB (disarankan 8 GB+ untuk pelatihan model).
- **Ruang Disk**: Minimal 2 GB ruang kosong.

---

### B. Kloning / Salin Proyek

#### Opsi 1: Menggunakan Git
```bash
git clone <URL_REPOSITORY_ANDA> sitrak-detect
cd sitrak-detect
```

#### Opsi 2: Salin Manual / Transfer via SCP/SFTP (Remote Server)
Bila menyalin proyek ke device/server remote tanpa Git:
```bash
# Contoh upload dari PC lokal ke server remote via SCP:
scp -r ./sitrak-detect user@192.168.1.100:/home/user/projects/
```

Masuk ke direktori proyek pada device target:
```bash
cd sitrak-detect
```

---

### C. Setup Virtual Environment

Selalu gunakan Virtual Environment agar dependensi terisolasi dari sistem utama.

#### Di Windows:
```powershell
# Menggunakan PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Jika ada kendala ExecutionPolicy pada PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

#### Di Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### D. Instalasi Dependensi Python

Setelah virtual environment aktif, jalankan pembaruan `pip` dan pasang paket yang tercantum di `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Paket utama yang terpasang meliputi:
- `fastapi` & `uvicorn[standard]` (HTTP Framework & ASGI Server)
- `ultralytics` & `torch` (Inference YOLO11 & YOLO26)
- `opencv-python-headless` (Pengolahan citra tanpa ketergantungan GUI OS)
- `numpy`, `pillow`, `pydantic`, `pytest`, `httpx`

---

### E. Catatan Khusus Dependensi OS (Linux / Ubuntu)

Jika memasang di server Linux headless (Ubuntu/Debian minimal), pastikan pustaka sistem pendukung image codec telah terpasang:

```bash
sudo apt-get update && sudo apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev
```

*(Catatan: Proyek ini menggunakan `opencv-python-headless` sehingga tidak membutuhkan pustaka X11/GUI penuh).*

---

### F. Akselerasi GPU NVIDIA (Opsional)

Secara default, instalasi `requirements.txt` akan memasang PyTorch versi CPU yang sudah sangat cepat untuk inferensi (hanya ~35-45 ms per foto pada prosesor modern).

Jika device target memiliki kartu grafis NVIDIA dan driver CUDA terpasang, Anda dapat mengaktifkan akselerasi GPU:

```bash
# Untuk CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Untuk CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verifikasi GPU terdeteksi:
```bash
python -c "import torch; print('CUDA Ready:', torch.cuda.is_available())"
```

---

### G. Verifikasi File Model Bobot

Pastikan folder `models/` berisi 3 file bobot model:
```bash
models/
├── bullet_hole_yolo.pt     # (~5.5 MB)
├── bullet_hole_yolo11.pt   # (~5.5 MB)
└── bullet_hole_yolo26.pt   # (~5.4 MB)
```

Jika file model belum tersedia pada instalasi baru, Anda dapat melatihnya langsung dengan menjalankan pipeline generator dan pelatihan (lihat [Bagian 9](#pipeline-pelatihan-ulang-model)).

---

### H. Konfigurasi Environment (`.env`)

1. Salin template `.env.example` menjadi `.env`:
   ```bash
   # Windows PowerShell:
   Copy-Item .env.example .env

   # Linux / macOS:
   cp .env.example .env
   ```

2. Buka dan sesuaikan file `.env`:
   ```env
   # Application Settings
   APP_NAME="Sitrak Shooting Target Detection Service"
   APP_ENV="development"
   DEBUG=true
   API_V1_STR="/api/v1"

   # Server Configuration
   HOST="0.0.0.0"       # 0.0.0.0 agar dapat diakses dari device lain di jaringan lokal
   PORT=8000

   # CORS Configuration (Daftar origin frontend yang diizinkan mengakses API)
   # Tambahkan IP laptop/device frontend Anda, atau gunakan "*" untuk mempermudah development
   CORS_ORIGINS=["http://localhost:3000","http://localhost:5173","http://127.0.0.1:3000","http://127.0.0.1:5173","*"]

   # Storage Configuration
   UPLOAD_DIR="static/uploads"
   ANNOTATED_DIR="static/annotated"
   MAX_UPLOAD_SIZE_MB=15

   # Vision Pipeline Settings
   TARGET_CANONICAL_SIZE=800
   DEFAULT_TARGET_TYPE="issf_standard_10_ring"
   CALIBRATION_MM_PER_PIXEL=0.35

   # AI Detection Settings
   DETECTION_ENGINE="hybrid"
   AI_CONFIDENCE_THRESHOLD=0.15
   AI_IOU_THRESHOLD=0.58
   ```

---

## Menjalankan Server

### 1. Mode Development
Gunakan mode ini saat menguji di komputer lokal atau saat proses integrasi:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server aktif dan siap menerima request:
- **API Base URL**: `http://localhost:8000`
- **Swagger UI Interaktif**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

### 2. Mode Production (Uvicorn Multi-Workers)
Gunakan mode ini di server produksi untuk menangani banyak request konkuren secara paralel:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --no-access-log
```

---

### 3. Mode Production Linux Service (Systemd)
Untuk server Linux (Ubuntu/Debian) agar service backend otomatis berjalan saat server dinyalakan (*auto-start on boot*):

1. Buat file service:
   ```bash
   sudo nano /etc/systemd/system/sitrak-detect.service
   ```

2. Masukkan konfigurasi berikut (sesuaikan path dan username Anda):
   ```ini
   [Unit]
   Description=Sitrak Shooting Target Detection Service
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/projects/sitrak-detect
   Environment="PATH=/home/ubuntu/projects/sitrak-detect/venv/bin"
   ExecStart=/home/ubuntu/projects/sitrak-detect/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

3. Aktifkan dan jalankan service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable sitrak-detect
   sudo systemctl start sitrak-detect
   sudo systemctl status sitrak-detect
   ```

---

## Akses Multi-Server & Remote Jaringan

Dalam arsitektur sistem Sitrak, aplikasi dapat terdiri dari beberapa server (misal: Server Frontend React/Vue di port 3000 dan Server Backend Deteksi di port 8000).

### 1. Akses Jaringan Lokal (LAN / Wi-Fi)
Karena server dijalankan dengan `HOST="0.0.0.0"`, device lain di jaringan Wi-Fi/LAN yang sama dapat mengakses API secara langsung menggunakan IP server:
- Cek IP server Anda:
  - Windows: `ipconfig` (cari *IPv4 Address*, contoh: `192.168.1.50`)
  - Linux/macOS: `ip a` atau `ifconfig`
- URL API dari device client: `http://192.168.1.50:8000/api/v1/detect`
- Pastikan firewall (Windows Defender Firewall / `ufw` di Linux) mengizinkan port `8000`:
  ```bash
  # Di Ubuntu/Debian:
  sudo ufw allow 8000/tcp
  ```

---

### 2. Akses Remote Server via SSH Tunnel
Bila server backend berada di cloud/VPS remote yang port 8000-nya tidak dibuka untuk publik, Anda dapat menghubungkan komputer lokal ke server melalui SSH Port Forwarding:
```bash
# Forward port 8000 server remote ke localhost:8000 di PC lokal
ssh -L 8000:localhost:8000 user@ip-server-remote
```
Setelah SSH terhubung, frontend di komputer Anda dapat memanggil API di `http://localhost:8000`.

---

### 3. Integrasi Frontend & Keamanan
> [!IMPORTANT]
> **Aturan Keamanan Frontend:**
> 1. **URL API backend HARUS diambil dari file `.env` frontend** (contoh: `VITE_API_URL=http://192.168.1.50:8000/api/v1/detect`), **JANGAN PERNAH disimpan di `localStorage` peramban**.
> 2. **Dilarang menyimpan API key, secret token, atau kredensial backend di `localStorage`**.

---

## Dokumentasi API & Contoh Request

### 1. Cek Kesehatan Sistem (`GET /api/v1/healthcheck`)
```bash
curl -X GET "http://localhost:8000/api/v1/healthcheck"
```
**Response (200 OK):**
```json
{
  "status": "healthy",
  "app_name": "Sitrak Shooting Target Detection Service",
  "version": "1.0.0",
  "storage_writable": true,
  "ai_detector_loaded": true,
  "ai_model_name": "bullet_hole_yolo.pt",
  "available_engines": ["yolo11", "yolo26", "model_sendiri", "cv"]
}
```

---

### 2. Upload Target Image (`POST /api/v1/detect`)
Mengunggah file foto sasaran tembak (fisik maupun digital) untuk dianalisis dan dinilai.

**Request via cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Accept: application/json" \
  -F "file=@/path/to/target_photo.jpg" \
  -F "engine=model_sendiri" \
  -F "caliber_mm=4.5" \
  -F "target_type=issf_standard_10_ring" \
  -F "shooter_name=Penembak 1"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "target_detected": true,
  "total_score": 96,
  "max_possible_score": 100,
  "accuracy_percentage": 96.0,
  "shots_count": 10,
  "shots": [
    {
      "id": 1,
      "x": 401.5,
      "y": 398.2,
      "score": 10,
      "decimal_score": 10.8,
      "distance_from_center": 2.45,
      "clock_direction": "12:00",
      "confidence": 0.89,
      "detection_source": "yolo_ai"
    },
    {
      "id": 2,
      "x": 374.0,
      "y": 446.0,
      "score": 9,
      "decimal_score": 9.3,
      "distance_from_center": 11.20,
      "clock_direction": "08:00",
      "confidence": 0.81,
      "detection_source": "yolo_ai"
    }
  ],
  "analytics": {
    "grouping_diameter_mm": 24.5,
    "directional_bias": "centered",
    "mean_point_of_impact_x": -0.8,
    "mean_point_of_impact_y": 1.2,
    "dispersion_radius_px": 35.0
  },
  "annotated_image_url": "/static/annotated/shot_result_abc123.jpg",
  "detection_engine_used": "model_sendiri",
  "metadata": {
    "perspective_corrected": true,
    "cropped_to_square": true,
    "crop_box": {
      "x": 0,
      "y": 40,
      "width": 788,
      "height": 788,
      "original_width": 788,
      "original_height": 860
    },
    "target_type": "issf_standard_10_ring",
    "caliber_mm": 4.5,
    "scale_mm_per_pixel": 0.35,
    "selected_engine": "model_sendiri"
  }
}
```

---

### 3. Stream Frame Detection (`POST /api/v1/detect/frame`)
Digunakan untuk pemrosesan berkala dari kamera webcam streaming front-end dalam format base64.

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

---

### 4. Status & Hot-Reload Model
- **Cek Status Model Aktif**:
  ```bash
  curl -X GET "http://localhost:8000/api/v1/detect/model-status"
  ```
- **Hot-Reload Model Baru (Tanpa Restart Server)**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/detect/model-reload"
  ```

---

## Pengujian & Verifikasi (Unit Tests)

Jalankan test suite untuk memastikan seluruh fungsi (validasi API, homography, auto-crop 1:1, dekomposisi perkenaan menempel, dan evaluasi model) berjalan 100%:

```bash
pytest tests/test_vision.py -v
```

Hasil uji: **14/14 tests PASSED (100%)**.

---

## Pipeline Pelatihan Ulang Model

Bila Anda memiliki koleksi foto sasaran baru atau ingin melatih model dari awal:

### 1. Bangun Dataset Final
Generator menggabungkan master scan lesan, foto smartphone pengguna, sampel sobekan fisik asli, serta **stiker tambalan sasaran (*target pasters*)** sebagai *hard negatives*:
```bash
python training/build_complete_target_dataset.py
```
Output tersimpan di `training/dataset_final/` (~974 gambar latih, 220 gambar validasi).

### 2. Latih Model YOLO11 & YOLO26
```bash
python training/train_final_models.py --epochs 8 --batch 16 --workers 2
```
Script akan:
1. Melatih model YOLO26 dan YOLO11.
2. Mengevaluasi performa model pada sasaran fisik asli bertambalan.
3. Memastikan tingkat deteksi stiker tambalan adalah `0 / 2` (100% diabaikan).
4. Otomatis mendeploy bobot model terbaik ke `models/bullet_hole_yolo.pt`, `models/bullet_hole_yolo11.pt`, dan `models/bullet_hole_yolo26.pt`.

---

## Lisensi & Dukungan

Proyek ini dikembangkan untuk kebutuhan analitik presisi sasaran tembak Sitrak. Untuk kendala teknis atau pengajuan fitur baru, silakan hubungi tim pengembang proyek Sitrak.
