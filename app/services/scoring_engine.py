import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from app.schemas.detection import ShotDetail, TargetAnalytics


class ScoringEngine:
    """
    Engine to compute shot scores, Euclidean distances, grouping dispersion,
    and shooter directional bias (MPI / compass / clock position).
    """

    def __init__(self, default_mm_per_pixel: float = 0.35):
        self.default_mm_per_pixel = default_mm_per_pixel

    def evaluate_shot(
        self,
        shot_id: int,
        x: float,
        y: float,
        center_x: float,
        center_y: float,
        ring_radii: Dict[int, float],
        bullet_radius_px: float = 6.0,
        mm_per_pixel: Optional[float] = None,
        confidence: Optional[float] = None,
        detection_source: str = "cv"
    ) -> ShotDetail:
        """
        Calculates distance from bullseye and ring score (1-10) for a given shot.
        Supports standard shooting rule where touching a higher ring line grants the higher score.
        """
        scale = mm_per_pixel or self.default_mm_per_pixel

        # Euclidean distance in pixels
        dx = x - center_x
        dy = y - center_y
        dist_px = math.hypot(dx, dy)
        dist_mm = dist_px * scale

        # Determine score: ring_radii maps score (10 to 1) -> outer radius in px
        # If the bullet touches the ring line, inward-gauging grants the higher point
        score = 0
        effective_dist = max(0.0, dist_px - (bullet_radius_px * 0.75))

        # Check rings from 10 down to 1
        sorted_rings = sorted(ring_radii.items(), key=lambda item: item[0], reverse=True)
        for ring_val, radius in sorted_rings:
            if effective_dist <= radius:
                score = ring_val
                break

        # Decimal score (ISSF 10th ring decimal score approximation)
        r10 = ring_radii.get(10, 20.0)
        r9 = ring_radii.get(9, 40.0)
        ring_step = max(1.0, r9 - r10)
        if dist_px <= r10:
            dec_score = round(10.9 - (dist_px / r10) * 0.9, 1)
        else:
            dec_score = max(0.0, round(10.0 - (dist_px - r10) / ring_step, 1))

        return ShotDetail(
            id=shot_id,
            x=round(x, 1),
            y=round(y, 1),
            score=score,
            distance_from_center=round(dist_mm, 2),
            decimal_score=dec_score,
            confidence=round(confidence, 2) if confidence is not None else None,
            detection_source=detection_source
        )

    def compute_analytics(
        self,
        shots: List[ShotDetail],
        center_x: float,
        center_y: float,
        mm_per_pixel: Optional[float] = None
    ) -> TargetAnalytics:
        """
        Calculates dispersion (grouping diameter) and directional bias based on
        shot distribution and Mean Point of Impact (MPI).
        """
        scale = mm_per_pixel or self.default_mm_per_pixel

        if not shots:
            return TargetAnalytics(
                grouping_diameter_mm=0.0,
                directional_bias="centered",
                mean_point_of_impact_x=0.0,
                mean_point_of_impact_y=0.0,
                dispersion_radius_px=0.0
            )

        coords = np.array([[s.x, s.y] for s in shots], dtype=np.float32)

        # 1. Grouping diameter (Extreme spread / Minimum Enclosing Circle)
        if len(shots) == 1:
            grouping_diam_px = 0.0
        elif len(shots) == 2:
            grouping_diam_px = float(np.linalg.norm(coords[0] - coords[1]))
        else:
            # Pairwise max distance (Extreme Spread standard)
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
            grouping_diam_px = float(np.max(dist_matrix))

        grouping_diam_mm = round(grouping_diam_px * scale, 1)

        # 2. Mean Point of Impact (MPI)
        mean_x = float(np.mean(coords[:, 0]))
        mean_y = float(np.mean(coords[:, 1]))
        offset_x = mean_x - center_x
        offset_y = mean_y - center_y  # In image coordinates, positive Y is down

        # 3. Directional Bias Calculation
        mpi_dist_px = math.hypot(offset_x, offset_y)
        bias_str = self._resolve_directional_bias(offset_x, offset_y, mpi_dist_px, scale)

        return TargetAnalytics(
            grouping_diameter_mm=grouping_diam_mm,
            directional_bias=bias_str,
            mean_point_of_impact_x=round(offset_x * scale, 2),
            mean_point_of_impact_y=round(offset_y * scale, 2),
            dispersion_radius_px=round(grouping_diam_px / 2.0, 1)
        )

    def _resolve_directional_bias(
        self,
        offset_x: float,
        offset_y: float,
        dist_px: float,
        scale: float
    ) -> str:
        """
        Maps vector (offset_x, offset_y) to compass & clock direction.
        offset_y is positive DOWN in image coordinate system.
        """
        # Threshold: if average offset is less than 3 mm, consider it centered
        if (dist_px * scale) < 4.0:
            return "centered"

        # Mathematical angle in degrees from center (0 = right, 90 = top, 180 = left, 270 = bottom)
        # Note: -offset_y because image Y is downwards
        angle_rad = math.atan2(-offset_y, offset_x)
        angle_deg = math.degrees(angle_rad) % 360

        # Compass & clock direction mapping
        if 22.5 <= angle_deg < 67.5:
            return "top-right"
        elif 67.5 <= angle_deg < 112.5:
            return "top"
        elif 112.5 <= angle_deg < 157.5:
            return "top-left"
        elif 157.5 <= angle_deg < 202.5:
            return "left"
        elif 202.5 <= angle_deg < 247.5:
            return "bottom-left"
        elif 247.5 <= angle_deg < 292.5:
            return "bottom"
        elif 292.5 <= angle_deg < 337.5:
            return "bottom-right"
        else:
            return "right"


    def _resolve_weapon_category(
        self,
        weapon_name: Optional[str] = None,
        weapon_type: Optional[str] = None,
        caliber_mm: Optional[float] = None
    ) -> str:
        """
        Identifies weapon family: 'pistol', 'senapan_serbu', 'sniper', 'smg', or 'senapan_angin'.
        """
        combined = f"{weapon_name or ''} {weapon_type or ''}".lower()
        cal = float(caliber_mm or 0)

        if any(k in combined for k in ["sniper", "spr", "runduk", "dmr", "awp", "remington", "7.62"]):
            return "sniper"
        if any(k in combined for k in ["pm2", "mp5", "smg", "submachine"]):
            return "smg"
        if any(k in combined for k in ["pistol", "g2", "glock", "sig", "revolver", "beretta", "colt", "makarov", "hs9"]):
            return "pistol"
        if any(k in combined for k in ["senapan", "ss1", "ss2", "m16", "m4", "ak47", "ak-", "serbu", "aug", "steyr", "fnc"]):
            return "senapan_serbu"
        if cal >= 8.5 and cal <= 11.5:
            return "pistol"
        if cal >= 5.0 and cal <= 7.7:
            return "senapan_serbu"
        if cal <= 4.6:
            return "senapan_angin"
        return "senapan_serbu"

    def generate_tactical_advice(
        self,
        bias_key: str,
        grouping_diameter_mm: float,
        accuracy_percentage: float,
        weapon_name: Optional[str] = None,
        weapon_type: Optional[str] = None,
        caliber_mm: Optional[float] = None,
        stance: Optional[str] = None,
        distance_m: Optional[float] = None
    ) -> List[str]:
        """
        Generates comprehensive, multi-layered tactical coaching advice tailored to:
        1. Biomechanical error identified by directional bias
        2. Weapon specific characteristics (Pistol G2/Glock vs SS2/M16 vs Sniper SPR vs SMG)
        3. Stance-specific posture (Tiarap, Berdiri, Berlutut)
        4. Remedial drill assignment for target training
        """
        norm_bias = (bias_key or "centered").lower().strip()
        category = self._resolve_weapon_category(weapon_name, weapon_type, caliber_mm)
        w_label = (weapon_name or weapon_type or "Senjata").strip()
        stance_str = (stance or "").lower()
        advice: List[str] = []

        # --- 1. BIOMECHANICAL & DIRECTIONAL CORRECTION ---
        if norm_bias in ["bottom-left", "condong ke kiri bawah"]:
            advice.append(
                "Koreksi Tarikan Picu (Trigger Jerking / Flinching): Titik perkenaan condong ke kiri bawah menandakan sentakan pelatuk mendadak dan pengetatan jari secara tiba-tiba sesaat sebelum tembakan meletus. Tarik picu secara halus, linier lurus ke belakang menggunakan bantalan ruas pertama jari telunjuk."
            )
            advice.append(
                "Stabilisasi Cengkeraman (Grip Pressure): Hindari gerakan meremas pegangan (milking grip) saat jari telunjuk bergerak menekan picu. Pertahankan tekanan cengkeraman tangan penembak tetap konstan dari awal pembidikan hingga peluru keluar dari laras."
            )
        elif norm_bias in ["bottom-right", "condong ke kanan bawah"]:
            advice.append(
                "Koreksi Tekanan Ibu Jari (Thumbing) & Pergelangan Tangan: Perkenaan kanan bawah mengindikasikan tekanan lateral ibu jari yang berlebihan pada frame senjata atau penurunan pergelangan tangan (breaking wrist down) saat menekan picu."
            )
            advice.append(
                "Kunci Sendi Pergelangan: Kunci sendi pergelangan tangan penembak pada posisi netral dan kokoh agar tidak mengendur di bawah gaya hentakan proyektil."
            )
        elif norm_bias in ["top-left", "condong ke kiri atas"]:
            advice.append(
                "Koreksi Antisipasi Hentakan (Pushing / Recoil Anticipation): Penembak terindikasi mendorong moncong senjata ke depan atau mengangkat bahu kiri tepat sebelum letupan untuk melawan hentakan."
            )
            advice.append(
                "Penyelarasan Pejera (Sight Alignment): Periksa celah pejera belakang; pastikan bilah pejera depan tidak menyembul ke kiri atas dan tetap berada di tengah rata (flush) dengan celah belakang."
            )
        elif norm_bias in ["top-right", "condong ke kanan atas"]:
            advice.append(
                "Koreksi Tekanan Tumit Tangan (Heeling): Perkenaan kanan atas terjadi akibat dorongan tumit telapak tangan kanan (heel of the palm) ke arah moncong saat mengantisipasi hentakan tembakan."
            )
            advice.append(
                "Relaksasi Otot Bahu & Pundak: Turunkan ketegangan pundak penembak; biarkan sistem rangka menyerap dorongan recoil secara alami tanpa dorongan otot aktif ke depan."
            )
        elif norm_bias in ["left", "condong ke kiri"]:
            advice.append(
                "Koreksi Kedalaman Jari pada Picu: Penempatan jari telunjuk terlalu dangkal (hanya di ujung kuku) menyebabkan tuas picu tertarik ke arah kiri penembak. Posisikan tengah bantalan ruas distal jari telunjuk tegak lurus 90 derajat terhadap picu."
            )
            advice.append(
                "Pemeriksaan Kemiringan Senjata (Canting): Pastikan senjata tidak miring ke kiri; kemiringan beberapa derajat akan menggeser lintasan balistik proyektil ke kiri sasaran."
            )
        elif norm_bias in ["right", "condong ke kanan"]:
            advice.append(
                "Koreksi Penarikan Samping (Trigger Snatching): Jari telunjuk terlalu dalam masuk ke rumah picu (hingga lipatan ruas kedua), sehingga saat picu ditarik senjata tertarik ke sisi kanan."
            )
            advice.append(
                "Cengkeraman Jari Pendukung: Kendurkan cengkraman tiga jari bawah (tengah, manis, kelingking) agar tidak memuntir poros senjata ke kanan saat tarikan picu."
            )
        elif norm_bias in ["top", "condong ke atas"]:
            advice.append(
                "Koreksi Pejera Naik (High Sight Picture): Pejera depan terlalu tinggi mencuat di atas celah pejera belakang atau fokus mata terbagi ke sasaran bukannya ke pejera depan."
            )
            advice.append(
                "Pengaturan Siklus Pernapasan: Lepas tembakan pada fase jeda pernapasan alami (natural respiratory pause) setelah menghembuskan separuh kapasitas paru-paru."
            )
        elif norm_bias in ["bottom", "condong ke bawah"]:
            advice.append(
                "Koreksi Penurunan Moncong (Dropping Muzzle): Moncong senjata turun sebelum peluru meninggalkan laras akibat pelepasan fokus bidikan atau kepala menunduk tergesa-gesa untuk melihat hasil tembakan."
            )
            advice.append(
                "Follow-Through (Lanjutan Bidikan): Tahan posisi bidikan, tekanan picu, dan pandangan mata ke pejera selama 1 detik setelah tembakan meletus sebelum mengendurkan senjata."
            )
        else:
            advice.append(
                "Konsistensi Titik Perkenaan Presisi: Pola perkenaan terpusat sangat baik di poros bullseye sasaran. Pertahankan kestabilan ritme penembakan dan visualisasi gambar bidik yang telah dicapai."
            )

        # --- 2. WEAPON-SPECIFIC COACHING ADVICE ---
        if category == "pistol":
            advice.append(
                f"Catatan Khusus Senjata [{w_label}]: Pada pistol kaliber pendek (sight radius pendek), penyimpangan pejera 1 mm di laras mengakibatkan deviasi besar pada lesan. Terapkan prinsip pembagian cengkeraman 60-70% pada tangan penopang (tangan kiri) dan 30-40% pada tangan penembak (tangan kanan). Manfaatkan trigger reset point tanpa melepaskan picu sepenuhnya ke depan untuk tembakan kedua yang cepat dan presisi."
            )
        elif category == "senapan_serbu":
            advice.append(
                f"Catatan Khusus Senjata [{w_label}]: Pastikan penempatan pipi pada popor (consistent cheek weld) selalu di titik yang identik untuk menjamin keselarasan eye-relief. Tanamkan popor senapan secara rapat di lekukan bahu (shoulder pocket) untuk meredam recoil kaliber 5.56mm/7.62mm secara simetris, dan tahan handguard tanpa menekan kotak magasen."
            )
        elif category == "sniper":
            advice.append(
                f"Catatan Khusus Senjata [{w_label}]: Hilangkan kesalahan paralaks teleskop optik (pastikan tidak ada bayangan melingkar di sekeliling reticle optik). Gunakan Natural Point of Aim (NPOA) dengan menyetel sandaran bipod/monopod agar senapan mengunci target tanpa ketegangan otot punggung atau leher."
            )
        elif category == "smg":
            advice.append(
                f"Catatan Khusus Senjata [{w_label}]: Pastikan popor terentang terkunci sempurna pada bahu penembak. Kendalikan muzzle climb yang cepat pada tembakan berturutan dengan tekanan ke bawah yang proporsional pada handguard depan."
            )
        elif category == "senapan_angin":
            advice.append(
                f"Catatan Khusus Senjata [{w_label}]: Proyektil memiliki waktu lintas laras (lock time) lebih lambat; terapkan follow-through yang lebih lama dan rileks tanpa sentakan recoil buatan."
            )

        # --- 3. STANCE-SPECIFIC GUIDANCE ---
        if "tiarap" in stance_str or "prone" in stance_str:
            advice.append(
                "Optimasi Sikap Tiarap (Prone Stance): Siku kedua tangan wajib membentuk segitiga piramida penyangga yang kokoh pada matras. Pastikan pernapasan menggunakan diafragma perut agar rongga dada tidak mengangkat moncong senapan saat membidik."
            )
        elif "berlutut" in stance_str or "kneeling" in stance_str:
            advice.append(
                "Optimasi Sikap Berlutut (Kneeling Stance): Pastikan siku tangan penopang bertumpu sedikit di depan tempurung lutut kiri (bukan tulang ketemu tulang) untuk menghindari slip mikro saat picu ditarik."
            )
        elif "berdiri" in stance_str or "standing" in stance_str:
            advice.append(
                "Optimasi Sikap Berdiri (Standing Stance): Buka kaki selebar bahu dengan condong badan agresif ke depan (athletic forward lean). Jangan melawan goyangan alami tubuh secara kaku; tarik picu bertahap saat lesan berada di zona goyangan terkecil."
            )

        # --- 4. GROUPING & DISPERSION EVALUATION ---
        if grouping_diameter_mm > 65.0:
            advice.append(
                f"Evaluasi Kerapatan (Grouping {grouping_diameter_mm} mm): Kelompok tembakan masih menyebar cukup luas. Prioritaskan latihan penembakan lambat terukur (slow fire) untuk memantapkan konsistensi mekanika tubuh penembak."
            )
        elif grouping_diameter_mm <= 35.0:
            advice.append(
                f"Evaluasi Kerapatan (Grouping {grouping_diameter_mm} mm): Kerapatan tembakan sangat luar biasa dan konsisten. Pertahankan disiplin bidik dan tingkatkan tempo penembakan secara bertahap."
            )

        # --- 5. ASSIGNED REMEDIAL DRILL ---
        if category == "pistol":
            if norm_bias in ["bottom-left", "bottom", "bottom-right"]:
                advice.append(
                    "Latihan Remedial yang Direkomendasikan: 'Ball & Dummy Drill' (campurkan peluru hampa/dummy secara acak di magasen) untuk melatih penembak agar tidak tersentak saat picu meletus, serta 'Wall Drill' (dry fire menempel di dinding)."
                )
            else:
                advice.append(
                    "Latihan Remedial yang Direkomendasikan: 'Coin Drill on Front Sight' (letakkan koin di atas pejera depan saat latihan kering; koin tidak boleh jatuh saat picu ditekan)."
                )
        elif category in ["senapan_serbu", "sniper"]:
            advice.append(
                "Latihan Remedial yang Direkomendasikan: 'Dry-Fire Reset Cadence Drill' (latihan kering fokus pada mendengar bunyi klik reset pelatuk tanpa merusak posisi cheek-weld dan gambar bidik di sasaran)."
            )

        return advice

    def generate_analysis_summary(
        self,
        bias_key: str,
        grouping_diameter_mm: float,
        accuracy_percentage: float,
        weapon_name: Optional[str] = None,
        shots_count: int = 0,
        total_score: int = 0,
        stance: Optional[str] = None
    ) -> str:
        """
        Builds a rich diagnostic summary paragraph describing shot distribution,
        clustering quality, and operational shooting readiness.
        """
        bias_trans = {
            "centered": "terpusat di tengah sasaran (bullseye)",
            "top-left": "condong ke sektor kiri atas",
            "top-right": "condong ke sektor kanan atas",
            "bottom-left": "condong ke sektor kiri bawah",
            "bottom-right": "condong ke sektor kanan bawah",
            "top": "condong ke sektor atas",
            "bottom": "condong ke sektor bawah",
            "left": "condong ke sektor kiri",
            "right": "condong ke sektor kanan"
        }
        bias_desc = bias_trans.get((bias_key or "centered").lower().strip(), "pada sasaran")
        w_text = f" menggunakan {weapon_name}" if weapon_name else ""
        s_text = f" posisi {stance}" if stance else ""

        if grouping_diameter_mm <= 30.0 and accuracy_percentage >= 85.0:
            cluster_text = f"Kerapatan tembakan sangat presisi (grouping {grouping_diameter_mm} mm) dengan perkenaan mayoritas menembus zona cincin 9 dan 10."
        elif grouping_diameter_mm <= 55.0:
            cluster_text = f"Kerapatan tembakan tergolong baik (grouping {grouping_diameter_mm} mm) dengan pengelompokan yang cukup solid."
        else:
            cluster_text = f"Sebaran tembakan masih cukup lebar (dispersi {grouping_diameter_mm} mm) dengan fluktuasi perkenaan antar tembakan."

        return (
            f"Hasil evaluasi {shots_count} butir tembakan{w_text}{s_text} menghasilkan total skor {total_score} "
            f"({accuracy_percentage}% akurasi). {cluster_text} Titik perkenaan rata-rata (MPI) teridentifikasi {bias_desc}."
        )


scoring_engine = ScoringEngine()

