import time
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np
from ultralytics import YOLO


def run_full_evaluation(
    dataset_yaml: str = "training/dataset_combined/data.yaml",
    master_image_path: str = "training/assets/kruger_master.webp",
    output_dir: str = "training/eval_results",
    imgsz: int = 416
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    models_to_test = [
        {"id": "yolo26", "name": "YOLO26", "path": "models/bullet_hole_yolo26.pt"},
        {"id": "yolo11", "name": "YOLO11", "path": "models/bullet_hole_yolo11.pt"},
        {"id": "default", "name": "Default Production Model", "path": "models/bullet_hole_yolo.pt"}
    ]

    results = {}
    master_img = cv2.imread(master_image_path)

    for m_info in models_to_test:
        mid = m_info["id"]
        mname = m_info["name"]
        mpath = Path(m_info["path"])

        if not mpath.exists():
            print(f"Skipping {mname}: weights file not found at {mpath}")
            continue

        print(f"\n{'='*65}")
        print(f" EVALUATING: {mname} ({mpath})")
        print(f"{'='*65}")

        model = YOLO(str(mpath))

        # 1. Validation metrics
        t_val_start = time.time()
        val_res = model.val(data=dataset_yaml, imgsz=imgsz, device="cpu", verbose=False)
        val_time = time.time() - t_val_start

        p = float(val_res.results_dict.get("metrics/precision(B)", 0))
        r = float(val_res.results_dict.get("metrics/recall(B)", 0))
        map50 = float(val_res.results_dict.get("metrics/mAP50(B)", 0))
        map95 = float(val_res.results_dict.get("metrics/mAP50-95(B)", 0))

        # 2. Latency benchmark on 30 inferences
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        # Warmup
        for _ in range(5):
            _ = model.predict(dummy, verbose=False)

        latencies = []
        for _ in range(25):
            t0 = time.perf_counter()
            _ = model.predict(dummy, verbose=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        avg_latency = float(np.mean(latencies))
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        # 3. Hard Negative Test on Clean Target (Must be 0 detections)
        hn_preds = model.predict(master_img, conf=0.25, verbose=False)
        hn_detections = len(hn_preds[0].boxes)

        # 4. Generate Visual Inspection Output
        # Run on a sample validation image to save visual proof of detection
        val_imgs = list(Path("training/dataset_kruger/images/val").glob("*.jpg"))
        if val_imgs:
            sample_val = cv2.imread(str(val_imgs[0]))
            preds_sample = model.predict(sample_val, conf=0.25, verbose=False)
            annotated_sample = sample_val.copy()
            for box in preds_sample[0].boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(annotated_sample, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
                cv2.putText(
                    annotated_sample,
                    f"{conf:.2f}",
                    (coords[0], max(15, coords[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
            vis_out_file = out_path / f"val_sample_{mid}.jpg"
            cv2.imwrite(str(vis_out_file), annotated_sample)

        results[mid] = {
            "name": mname,
            "path": str(mpath),
            "precision": p,
            "recall": r,
            "map50": map50,
            "map95": map95,
            "latency_ms": avg_latency,
            "fps": fps,
            "hard_negative_passed": (hn_detections == 0),
            "false_positive_count": hn_detections
        }

        print(f"Results for {mname}:")
        print(f"  Precision:       {p*100:.2f}%")
        print(f"  Recall:          {r*100:.2f}%")
        print(f"  mAP@50:          {map50*100:.2f}%")
        print(f"  mAP@50-95:       {map95*100:.2f}%")
        print(f"  CPU Latency:     {avg_latency:.2f} ms ({fps:.1f} FPS)")
        print(f"  Hard Negative:   {'PASSED (0 false positives)' if hn_detections == 0 else f'FAILED ({hn_detections} false positives)'}")

    # Generate Markdown Report
    report_file = out_path / "EVALUATION_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Laporan Evaluasi Deteksi Sasaran Tembak Krüger\n\n")
        f.write(f"Tanggal Pengujian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Ringkasan Performa Model\n\n")
        f.write("| Model | Presisi | Recall | mAP@50 | mAP@50-95 | Latensi CPU | Status Hard Negative |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for mid, r in results.items():
            hn_str = "LULUS (0 FP)" if r["hard_negative_passed"] else f"GAGAL ({r['false_positive_count']} FP)"
            f.write(f"| **{r['name']}** | {r['precision']*100:.1f}% | {r['recall']*100:.1f}% | {r['map50']*100:.1f}% | {r['map95']*100:.1f}% | {r['latency_ms']:.1f} ms | {hn_str} |\n")
        f.write("\n## Catatan Evaluasi\n\n")
        f.write("- **YOLO26**: Optimal untuk lubang berukuran kecil dengan arsitektur Small-Target-Aware Label Assignment (STAL).\n")
        f.write("- **YOLO11**: Arsitektur Ultralytics terbaru dengan akurasi dan lokalisasi spasial tinggi.\n")
        f.write("- **Uji Hard Negative**: Dilakukan pada gambar sasaran Krüger polos tanpa tembakan untuk memastikan angka 1-10 dan teks tidak memicu false positive.\n")

    print(f"\n[SUCCESS] Evaluation report and visual artifacts generated at: {report_file.resolve()}")
    return results


if __name__ == "__main__":
    run_full_evaluation()
