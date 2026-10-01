"""
Bike Fitting Analyzer  (Python 3 + DearPyGui + OpenCV + MediaPipe Pose)

Input : webcam atau file video, tampak SAMPING / DEPAN / BELAKANG pesepeda
Output: titik yang perlu diperbaiki, rekomendasi, dan skor 0-100

Install : pip install dearpygui opencv-python mediapipe numpy
Jalankan: python bikefit.py
"""
import os
import time
import urllib.request
from collections import defaultdict, deque

import cv2
import numpy as np
import mediapipe as mp
import dearpygui.dearpygui as dpg
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Mengunduh model pose (sekali saja, ~9 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# --------------------------------------------------------------------------
# Definisi parameter fitting (rentang ideal = panduan umum, bukan mutlak)
# agg: cara merangkum nilai selama beberapa detik terakhir
# --------------------------------------------------------------------------
BIKE_TYPES = ["Road", "MTB", "City/Hybrid"]
VIEWS = ["Samping", "Depan", "Belakang"]


def M(key, label, weight, ideal, low, high, agg="median", unit="°", fmt=".0f", tol=15.0):
    if not isinstance(ideal, dict):
        ideal = {b: ideal for b in BIKE_TYPES}
    return dict(key=key, label=label, weight=weight, ideal=ideal, low=low, high=high,
                agg=agg, unit=unit, fmt=fmt, tol=tol)


SIDE_METRICS = [
    M("knee", "Sudut lutut (BDC)", 0.30,
      {"Road": (140, 150), "MTB": (138, 150), "City/Hybrid": (135, 150)},
      "Sadel terlalu RENDAH (lutut terlalu menekuk di titik terbawah). "
      "Naikkan sadel bertahap 3-5 mm lalu ukur ulang.",
      "Sadel terlalu TINGGI (lutut hampir lurus, panggul bisa bergoyang). "
      "Turunkan sadel bertahap 3-5 mm lalu ukur ulang.", agg="p95"),
    M("hip", "Sudut pinggul (tutup)", 0.20,
      {"Road": (45, 65), "MTB": (50, 75), "City/Hybrid": (60, 90)},
      "Pinggul terlalu tertutup: torso terlalu rendah/sadel terlalu maju. "
      "Naikkan stem/spacer atau geser sadel sedikit ke belakang.",
      "Pinggul terlalu terbuka: posisi terlalu tegak/cockpit terlalu dekat. "
      "Turunkan stem atau gunakan stem lebih panjang.", agg="p5"),
    M("torso", "Sudut punggung (thd horizontal)", 0.20,
      {"Road": (35, 50), "MTB": (45, 60), "City/Hybrid": (60, 80)},
      "Punggung terlalu rata/agresif. Naikkan handlebar atau pakai stem lebih pendek/naik.",
      "Punggung terlalu tegak. Turunkan handlebar atau pakai stem lebih panjang."),
    M("elbow", "Sudut siku", 0.15,
      {"Road": (150, 170), "MTB": (150, 170), "City/Hybrid": (155, 175)},
      "Siku terlalu menekuk (jangkauan pendek). Pakai stem lebih panjang atau geser sadel ke belakang.",
      "Siku hampir terkunci (jangkauan terlalu jauh). Pakai stem lebih pendek atau geser sadel ke depan."),
    M("shoulder", "Sudut bahu (lengan-torso)", 0.15,
      {"Road": (75, 95), "MTB": (75, 95), "City/Hybrid": (60, 90)},
      "Lengan terlalu rapat ke badan / reach pendek. Panjangkan stem atau periksa lebar handlebar.",
      "Lengan terlalu terentang ke depan. Pendekkan stem atau naikkan handlebar."),
]

KNEE_LOW = ("Lutut mengarah ke LUAR (varus). Periksa posisi dan rotasi cleat, "
            "dan pertimbangkan stance lebih sempit (kurangi spacer pedal).")
KNEE_HIGH = ("Lutut jatuh ke DALAM (valgus). Periksa posisi/rotasi cleat, lebarkan stance "
             "(spacer pedal), cek penyangga lengkung kaki (insole/wedge), dan pastikan sadel tidak terlalu tinggi.")
PELVIS_HIGH = ("Panggul bergoyang/turun sebelah saat mengayuh: sadel kemungkinan terlalu tinggi, "
               "atau ada perbedaan panjang kaki/fleksibilitas. Turunkan sadel 2-3 mm, cek shim cleat & tilt sadel.")
TRUNK_HIGH = ("Torso banyak bergoyang ke samping: cek sadel terlalu tinggi/jauh, reach terlalu panjang, "
              "atau kekuatan core. Coba turunkan sadel/pendekkan reach.")
SHTILT_HIGH = ("Bahu tidak rata: periksa handlebar/hoods sejajar, tinggi sadel & tilt simetris, "
               "dan kemungkinan asimetri tubuh. Pastikan kamera tidak miring saat merekam.")

FRONT_COMMON = [
    ("knee_l", "Lutut kiri (tracking)", -6, 6, "extreme", KNEE_LOW, KNEE_HIGH, 10.0),
    ("knee_r", "Lutut kanan (tracking)", -6, 6, "extreme", KNEE_LOW, KNEE_HIGH, 10.0),
    ("pelvis", "Goyangan panggul", 0, 5, "range", "", PELVIS_HIGH, 10.0),
    ("trunk", "Goyangan torso", 0, 6, "range", "", TRUNK_HIGH, 10.0),
    ("sh_tilt", "Kemiringan bahu", 0, 4, "absmedian", "", SHTILT_HIGH, 8.0),
]
BAR_METRIC = M("bar", "Lebar setang / bahu", 0.15, (0.90, 1.20),
               "Setang tampak lebih sempit dari bahu: pertimbangkan setang lebih lebar "
               "agar dada terbuka dan napas lebih lega.",
               "Setang tampak lebih lebar dari bahu: pertimbangkan setang lebih sempit "
               "(lebih aerodinamis, bahu lebih rileks).", unit="x", fmt=".2f", tol=0.25)


def frontal_metrics(with_bar):
    weights = {"knee_l": .20, "knee_r": .20, "pelvis": .20, "trunk": .15, "sh_tilt": .10}
    ms = [M(k, lab, weights[k], (lo, hi), lo_t, hi_t, agg=agg, tol=tol)
          for k, lab, lo, hi, agg, lo_t, hi_t, tol in FRONT_COMMON]
    return ms + ([BAR_METRIC] if with_bar else [])


METRICS_BY_VIEW = {"Samping": SIDE_METRICS,
                   "Depan": frontal_metrics(True),
                   "Belakang": frontal_metrics(False)}

VIEW_TIPS = {
    "Samping": "Kamera tegak lurus dari samping setinggi pinggul; seluruh tubuh & sepeda terlihat. "
               "Pilih sisi tubuh yang menghadap kamera.",
    "Depan": "Kamera di depan sepeda setinggi pinggul, lurus ke depan (tidak miring). Kedua tangan, "
             "bahu, pinggul, lutut & kaki harus terlihat. Kiri/kanan = sisi tubuh pesepeda.",
    "Belakang": "Kamera di belakang sepeda setinggi pinggul, lurus (tidak miring). Bahu, pinggul, "
                "lutut & kaki harus terlihat. Kiri/kanan = sisi tubuh pesepeda.",
}


# --------------------------------------------------------------------------
# Fungsi geometri & statistik
# --------------------------------------------------------------------------
def angle3(a, b, c):
    """Sudut interior di titik b (derajat) untuk titik a-b-c."""
    ba, bc = np.array(a) - np.array(b), np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def angle_horizontal(p_from, p_to):
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    return float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-9)))


def tilt(p, q):
    """Kemiringan garis p->q terhadap horizontal, -90..90 derajat."""
    a = float(np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])))
    return a - 180 if a > 90 else a + 180 if a < -90 else a


def aggregate(agg, data):
    a = np.asarray(data, float)
    p95, p5 = np.percentile(a, 95), np.percentile(a, 5)
    if agg == "p95":
        return p95
    if agg == "p5":
        return p5
    if agg == "range":
        return p95 - p5
    if agg == "absmedian":
        return abs(np.median(a))
    if agg == "extreme":
        return p95 if abs(p95) >= abs(p5) else p5
    return np.median(a)


def score_metric(value, lo, hi, tol):
    dev = max(lo - value, value - hi, 0.0)
    return max(0.0, 100.0 * (1.0 - dev / tol))


def get_pts(res, w, h, idx):
    lms = res.pose_landmarks[0]
    pts, vmin = {}, 1.0
    for name, i in idx.items():
        lm = lms[i]
        pts[name] = (lm.x * w, lm.y * h)
        vmin = min(vmin, lm.visibility or 0.0)
    return pts, vmin


def draw_chain(frame, pts, chain, color):
    for a, b in zip(chain[:-1], chain[1:]):
        cv2.line(frame, tuple(map(int, pts[a])), tuple(map(int, pts[b])), color, 3)


# --------------------------------------------------------------------------
# Aplikasi
# --------------------------------------------------------------------------
class BikeFitApp:
    W, H = 960, 540  # resolusi tekstur internal (16:9); ditampilkan dengan skala dinamis

    def __init__(self):
        ensure_model()
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose = vision.PoseLandmarker.create_from_options(opts)
        self.ts_ms = 0  # timestamp harus selalu naik
        self.cap = None
        self.is_file = False
        self.paused = False
        self.fps = 30.0
        self.last_t = 0.0
        self.frame_i = 0
        self.view = "Samping"
        self.hist = defaultdict(lambda: deque(maxlen=150))
        self.result = None
        self.loop_count = 0     # jumlah putaran ulang video
        self.stable = 0         # evaluasi berturut-turut yang bukan "Perhatikan"
        self.settled = False    # True jika hasil sudah pasti
        self.prev_pass_vals = None   # nilai parameter di akhir putaran sebelumnya
        self.converged_passes = 0    # putaran berturut-turut dengan nilai tidak berubah
        self.status_msg = "Pilih sumber: webcam atau file video."
        self.proc_fps = 0.0
        self._t_prev = time.time()
        self.src_name = "Sumber: -"
        self._rec_sig = None

    @property
    def metrics(self):
        return METRICS_BY_VIEW[self.view]

    # ---------------- sumber video ----------------
    def open_source(self, src, is_file):
        self.close_source()
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.status_msg = "Gagal membuka sumber video."
            self.refresh_status()
            return
        self.cap, self.is_file = cap, is_file
        self.src_name = (os.path.basename(str(src)) if is_file else f"Webcam {src}")
        self.paused = False
        dpg.configure_item("btn_pause", label="Jeda")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.paused = False
        self.reset()
        self.status_msg = "Menganalisis... (kayuh stabil min. 5-10 detik)"
        self.refresh_status()

    def close_source(self):
        if self.cap:
            self.cap.release()
        self.cap = None

    def reset(self):
        self.hist.clear()
        self.result = None
        self.frame_i = 0
        self.loop_count = 0
        self.stable = 0
        self.settled = False
        self.prev_pass_vals = None
        self.converged_passes = 0

    def set_view(self, view):
        self.view = view
        self.reset()
        rebuild_table(self.metrics)
        self.clear_results()
        dpg.configure_item("side_grp", show=(view == "Samping"))
        dpg.set_value("tips", VIEW_TIPS[view])

    # ---------------- pemrosesan ----------------
    def step(self):
        if self.cap is None or self.paused:
            return
        now = time.time()
        if self.is_file and now - self.last_t < 1.0 / self.fps:
            return
        self.last_t = now
        ok, frame = self.cap.read()
        if not ok:
            self.evaluate()
            self.track_settle()
            self.check_converged()
            max_loops = int(dpg.get_value("max_loops"))
            looping = self.is_file and dpg.get_value("loop_on")
            if (looping and not self.settled and self.converged_passes < 2
                    and self.loop_count < max_loops):
                self.loop_count += 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.status_msg = (f"Putaran {self.loop_count}/{max_loops}: hasil masih "
                                   f"'Perhatikan' / data belum cukup, video diulang...")
                self.update_ui()
                return
            self.close_source()
            if self.settled:
                self.status_msg = ("Video selesai. Semua parameter sudah OK/Perbaiki "
                                   "(tidak ada yang tertahan di 'Perhatikan').")
            elif self.converged_passes >= 2:
                names = ", ".join(self.attention_names()) or "-"
                self.status_msg = (f"Video selesai. Nilai sudah konsisten antar putaran; "
                                   f"parameter berikut memang berada di zona 'Perhatikan' "
                                   f"(bukan noise): {names}. Ikuti rekomendasi lalu rekam ulang.")
            elif looping:
                self.status_msg = (f"Batas {max_loops} putaran tercapai; hasil masih "
                                   f"'Perhatikan' atau data belum cukup.")
            else:
                self.status_msg = "Video selesai. Hasil akhir ditampilkan."
            self.update_ui()
            return
        self.frame_i += 1
        t = time.time()
        dt, self._t_prev = t - self._t_prev, t
        if 0 < dt < 1:
            self.proc_fps = 0.9 * self.proc_fps + 0.1 / dt
        frame = self.process(frame)
        self.show(frame)
        if self.frame_i % 10 == 0:
            self.evaluate()
            self.track_settle()
            self.update_ui()

    def attention_names(self):
        r = self.result
        if not r:
            return []
        return [row["m"]["label"] for row in r["rows"]
                if self.status_of(row["score"])[0] == "Perhatikan"]

    def has_attention(self):
        """True jika kesimpulan total ATAU salah satu parameter masih 'Perhatikan'."""
        r = self.result
        return (self.status_of(r["total"])[0] == "Perhatikan") or bool(self.attention_names())

    def check_converged(self):
        """Bandingkan nilai akhir putaran ini dengan putaran sebelumnya."""
        r = self.result
        if not r:
            self.prev_pass_vals, self.converged_passes = None, 0
            return
        vals = [(row["value"], row["m"]["tol"] * 0.07) for row in r["rows"]]
        prev = self.prev_pass_vals
        if prev and len(prev) == len(vals) and all(abs(a - b[0]) <= b[1] for b, (a, _) in
                                                   zip(vals, prev)):
            self.converged_passes += 1
        else:
            self.converged_passes = 0
        self.prev_pass_vals = [(v, t) for v, t in vals]

    def track_settle(self):
        """Hasil pasti jika 3 evaluasi berturut-turut tidak ada yang berstatus 'Perhatikan'."""
        r = self.result
        if not r:
            self.stable = 0
            return
        if not self.has_attention():
            self.stable += 1
            if self.stable >= 3 and not self.settled:
                self.settled = True
                self.status_msg = ("Semua parameter sudah OK/Perbaiki; video dituntaskan "
                                   "sampai akhir putaran ini lalu berhenti.")
        else:
            self.stable = 0

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.ts_ms += 34
        res = self.pose.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self.ts_ms)
        if not res.pose_landmarks:
            cv2.putText(frame, "Pose tidak terdeteksi", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        vals = (self.side_values(res, frame, w, h) if self.view == "Samping"
                else self.front_values(res, frame, w, h))
        if vals is None:
            return frame
        for k, v in vals.items():
            self.hist[k].append(v)
        y = 30
        for k, v in vals.items():
            cv2.putText(frame, f"{k}: {v:.2f}" if abs(v) < 5 and k == "bar" else f"{k}: {v:.0f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y += 28
        return frame

    @staticmethod
    def warn(frame, msg):
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    def side_values(self, res, frame, w, h):
        off = 0 if dpg.get_value("side") == "Kiri" else 1  # indeks kanan = kiri + 1
        base = {"SHOULDER": 11, "ELBOW": 13, "WRIST": 15, "HIP": 23, "KNEE": 25, "ANKLE": 27}
        pts, vmin = get_pts(res, w, h, {n: i + off for n, i in base.items()})
        draw_chain(frame, pts, list(base), (0, 255, 0) if vmin > 0.5 else (0, 165, 255))
        for p in pts.values():
            cv2.circle(frame, tuple(map(int, p)), 6, (0, 0, 255), -1)
        if vmin < 0.5:
            self.warn(frame, "Titik tubuh kurang jelas / sisi salah?")
            return None
        return {
            "knee": angle3(pts["HIP"], pts["KNEE"], pts["ANKLE"]),
            "hip": angle3(pts["SHOULDER"], pts["HIP"], pts["KNEE"]),
            "torso": angle_horizontal(pts["HIP"], pts["SHOULDER"]),
            "elbow": angle3(pts["SHOULDER"], pts["ELBOW"], pts["WRIST"]),
            "shoulder": angle3(pts["HIP"], pts["SHOULDER"], pts["ELBOW"]),
        }

    def front_values(self, res, frame, w, h):
        front = self.view == "Depan"
        idx = {"SL": 11, "SR": 12, "HL": 23, "HR": 24, "KL": 25, "KR": 26, "AL": 27, "AR": 28}
        if front:
            idx.update({"WL": 15, "WR": 16})
        pts, vmin = get_pts(res, w, h, idx)
        col = (0, 255, 0) if vmin > 0.5 else (0, 165, 255)
        for chain in (["SL", "SR"], ["HL", "HR"], ["SL", "HL"], ["SR", "HR"],
                      ["HL", "KL", "AL"], ["HR", "KR", "AR"]):
            draw_chain(frame, pts, chain, col)
        if front:
            draw_chain(frame, pts, ["WL", "WR"], (255, 0, 255))
        for p in pts.values():
            cv2.circle(frame, tuple(map(int, p)), 6, (0, 0, 255), -1)
        if vmin < 0.5:
            self.warn(frame, "Ada titik tubuh yang belum terlihat jelas")
            return None
        vals = {}
        mid_x = (pts["HL"][0] + pts["HR"][0]) / 2
        for s in "LR":
            hip, knee, ank = pts["H" + s], pts["K" + s], pts["A" + s]
            t = (knee[1] - hip[1]) / ((ank[1] - hip[1]) or 1e-9)
            line_x = hip[0] + t * (ank[0] - hip[0])
            inward = (knee[0] - line_x) * (1.0 if mid_x >= hip[0] else -1.0)  # + = ke dalam
            leg = np.hypot(ank[0] - hip[0], ank[1] - hip[1])
            vals["knee_" + s.lower()] = float(np.degrees(np.arctan2(inward, 0.5 * leg)))
        vals["pelvis"] = tilt(pts["HL"], pts["HR"])
        vals["sh_tilt"] = tilt(pts["SL"], pts["SR"])
        sh = np.mean([pts["SL"], pts["SR"]], axis=0)
        hp = np.mean([pts["HL"], pts["HR"]], axis=0)
        vals["trunk"] = float(np.degrees(np.arctan2(sh[0] - hp[0], hp[1] - sh[1])))
        if front:
            vals["bar"] = abs(pts["WL"][0] - pts["WR"][0]) / (abs(pts["SL"][0] - pts["SR"][0]) + 1e-9)
        return vals

    def evaluate(self):
        ms = self.metrics
        if len(self.hist[ms[0]["key"]]) < 30:
            return
        bike = dpg.get_value("bike")
        rows, total, wsum = [], 0.0, 0.0
        for m in ms:
            lo, hi = m["ideal"][bike]
            v = float(aggregate(m["agg"], self.hist[m["key"]]))
            s = score_metric(v, lo, hi, m["tol"])
            total += s * m["weight"]
            wsum += m["weight"]
            tip = m["low"] if v < lo else m["high"] if v > hi else None
            rows.append(dict(m=m, value=v, lo=lo, hi=hi, score=s, tip=tip))
        self.result = dict(rows=rows, total=total / wsum, bike=bike, view=self.view)

    # ---------------- tampilan ----------------
    def show(self, frame):
        h, w = frame.shape[:2]
        s = min(self.W / w, self.H / h)
        nw, nh = int(w * s), int(h * s)
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        x0, y0 = (self.W - nw) // 2, (self.H - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(frame, (nw, nh))
        rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        dpg.set_value("tex", rgba.ravel())

    def show_placeholder(self):
        frame = np.full((self.H, self.W, 3), (38, 33, 30), np.uint8)  # BGR
        txt = "Pilih Webcam atau Buka Video"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, txt, ((self.W - tw) // 2, (self.H + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (190, 185, 175), 2, cv2.LINE_AA)
        self.show(frame)

    @staticmethod
    def status_of(score):
        if score >= 85:
            return "OK", C_OK
        if score >= 60:
            return "Perhatikan", C_WARN
        return "Perbaiki", C_BAD

    # ---------------- aksi tombol ----------------
    def stop(self):
        self.close_source()
        self.proc_fps = 0.0
        self.status_msg = "Sumber dihentikan."
        self.refresh_status()

    def toggle_pause(self):
        if self.cap is None:
            return
        self.paused = not self.paused
        dpg.configure_item("btn_pause", label="Lanjut" if self.paused else "Jeda")
        self.status_msg = "Dijeda." if self.paused else "Menganalisis..."
        self.refresh_status()

    def reset_all(self):
        self.reset()
        self.clear_results()
        self.status_msg = "Data direset."
        self.refresh_status()

    # ---------------- pembaruan UI ----------------
    def refresh_status(self):
        msg, low = self.status_msg, self.status_msg.lower()
        if "gagal" in low:
            col = C_BAD
        elif "semua parameter" in low or "sudah pasti" in low:
            col = C_OK
        elif "perhatikan" in low or "batas" in low or "belum" in low:
            col = C_WARN
        else:
            col = C_INFO
        dpg.set_value("status", msg)
        dpg.configure_item("status", color=col)

    def clear_results(self):
        dpg.set_value("overall", "--")
        dpg.configure_item("overall", color=C_MUTED)
        dpg.set_value("overall_status", "Belum ada hasil")
        dpg.configure_item("overall_status", color=C_MUTED)
        dpg.set_value("overall_ctx", f"{self.view} | {dpg.get_value('bike')}")
        dpg.set_value("overall_bar", 0.0)
        dpg.configure_item("overall_bar", overlay="-")
        dpg.bind_item_theme("overall_bar", "th_idle")
        self._rec_sig = None
        dpg.delete_item("rec_group", children_only=True)
        dpg.add_text("Rekomendasi muncul setelah data cukup (sekitar 30 frame).",
                     parent="rec_group", wrap=0, color=C_MUTED)

    def render_recs(self, rows):
        tips = sorted([r for r in rows if r["tip"]], key=lambda x: x["score"])
        sig = tuple((r["m"]["key"], r["tip"], self.status_of(r["score"])[0]) for r in tips)
        if sig == self._rec_sig:  # hindari rebuild (dan loncatan scroll) bila tidak berubah
            return
        self._rec_sig = sig
        dpg.delete_item("rec_group", children_only=True)
        if not tips:
            dpg.add_text("Semua parameter dalam rentang ideal. Posisi sudah baik.",
                         parent="rec_group", wrap=0, color=C_OK)
            return
        for i, r in enumerate(tips):
            st, col = self.status_of(r["score"])
            if i:
                dpg.add_separator(parent="rec_group")
            dpg.add_text(f"{r['m']['label']}  [{st}]", parent="rec_group", color=col)
            dpg.add_text(r["tip"], parent="rec_group", wrap=0, indent=12)

    def update_ui(self):
        self.refresh_status()
        if self.cap:
            dpg.set_value("src_txt", f"{self.src_name}  |  {self.proc_fps:.0f} FPS")
        dpg.set_value("loop_info", f"{self.loop_count} / {int(dpg.get_value('max_loops'))}"
                      if self.is_file else "Hanya file video")
        r = self.result
        if not r or r["view"] != self.view:
            return
        total = r["total"]
        label, col = self.status_of(total)
        dpg.set_value("overall", f"{total:.0f}")
        dpg.configure_item("overall", color=col)
        dpg.set_value("overall_status", label)
        dpg.configure_item("overall_status", color=col)
        dpg.set_value("overall_ctx", f"{r['view']} | {r['bike']}")
        dpg.set_value("overall_bar", total / 100.0)
        dpg.configure_item("overall_bar", overlay=f"{total:.0f} / 100")
        dpg.bind_item_theme("overall_bar", STATUS_THEME[label])
        for row in r["rows"]:
            m, k = row["m"], row["m"]["key"]
            f, u = m["fmt"], m["unit"]
            v, lo, hi = row["value"], row["lo"], row["hi"]
            st, c = self.status_of(row["score"])
            dpg.set_value(f"{k}_val", f"{v:{f}}{u}")
            dpg.set_value(f"{k}_ideal", f"{lo:{f}} - {hi:{f}}{u}")
            dpg.set_value(f"{k}_bar", row["score"] / 100.0)
            dpg.configure_item(f"{k}_bar", overlay=f"{row['score']:.0f}")
            dpg.bind_item_theme(f"{k}_bar", STATUS_THEME[st])
            dpg.set_value(f"{k}_st", st)
            dpg.configure_item(f"{k}_st", color=c)
        self.render_recs(r["rows"])

    def save_report(self):
        if not self.result:
            self.status_msg = "Belum ada hasil untuk disimpan."
            self.refresh_status()
            return
        r = self.result
        lines = [f"LAPORAN BIKE FITTING - {time.strftime('%Y-%m-%d %H:%M')}",
                 f"Sudut pandang: {r['view']} | Tipe sepeda: {r['bike']}",
                 f"Skor total: {r['total']:.0f}/100", ""]
        for row in r["rows"]:
            m = row["m"]
            lines.append(f"{m['label']}: {row['value']:{m['fmt']}}{m['unit']} "
                         f"(ideal {row['lo']:{m['fmt']}} s/d {row['hi']:{m['fmt']}}{m['unit']}) "
                         f"skor {row['score']:.0f} [{self.status_of(row['score'])[0]}]")
        lines += ["", "REKOMENDASI:"]
        lines += [f"- {row['m']['label']}: {row['tip']}" for row in r["rows"] if row["tip"]] \
            or ["- Tidak ada, semua parameter ideal."]
        name = f"laporan_bikefit_{r['view'].lower()}.txt"
        with open(name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.status_msg = f"Laporan disimpan: {name}"
        self.refresh_status()


# --------------------------------------------------------------------------
# Konstanta UI
# --------------------------------------------------------------------------
C_OK, C_WARN, C_BAD = (80, 220, 100), (255, 200, 60), (255, 90, 90)
C_INFO, C_MUTED = (120, 180, 255), (150, 155, 165)
STATUS_THEME = {"OK": "th_ok", "Perhatikan": "th_warn", "Perbaiki": "th_bad"}

PAD, GAP = 8, 8           # padding jendela & jarak antar panel (px), sesuai default ImGui
HEADER_H = 22             # tinggi header (hasil kalibrasi gaya default)
LEFT_RATIO = 0.58         # proporsi lebar panel video; sisanya panel hasil
CONTROLS_H = 296          # tinggi area kontrol di bawah video (hasil kalibrasi)
FOOTER_H = 72             # tinggi footer panel hasil (tombol simpan + catatan)
ASPECT = BikeFitApp.W / BikeFitApp.H

DESC = {
    "knee": "Sudut pinggul-lutut-pergelangan kaki saat lutut paling lurus (BDC). Menentukan tinggi sadel.",
    "hip": "Sudut bahu-pinggul-lutut saat paling tertutup (puncak kayuhan).",
    "torso": "Kemiringan garis pinggul-bahu terhadap horizontal.",
    "elbow": "Sudut bahu-siku-pergelangan tangan. 180° berarti lengan lurus penuh.",
    "shoulder": "Sudut pinggul-bahu-siku (lengan terhadap badan).",
    "knee_l": "Simpangan lutut kiri dari garis pinggul-kaki. Plus = ke dalam (valgus), minus = ke luar.",
    "knee_r": "Simpangan lutut kanan dari garis pinggul-kaki. Plus = ke dalam (valgus), minus = ke luar.",
    "pelvis": "Besar goyangan panggul (rentang kemiringan) selama mengayuh.",
    "trunk": "Besar goyangan torso ke samping selama mengayuh.",
    "sh_tilt": "Kemiringan garis bahu terhadap horizontal. Kamera miring ikut memengaruhi.",
    "bar": "Jarak pergelangan tangan dibanding lebar bahu (pendekatan lebar setang).",
}


# --------------------------------------------------------------------------
# Tema, font, layout
# --------------------------------------------------------------------------
def make_status_themes():
    """Hanya warna isi progress bar per status. Gaya lain tetap bawaan ImGui/DearPyGui."""
    for tag, rgb in (("th_ok", (60, 190, 90)), ("th_warn", (230, 170, 40)),
                     ("th_bad", (220, 70, 70)), ("th_idle", (90, 96, 110))):
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvProgressBar):
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, rgb,
                                    category=dpg.mvThemeCat_Core)


def layout():
    """Hitung ulang ukuran semua panel agar mengikuti ukuran jendela."""
    cw, ch = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
    inner_w = cw - 2 * PAD
    body_h = max(300, ch - 2 * PAD - HEADER_H - GAP)
    left_w = int((inner_w - GAP) * LEFT_RATIO)
    right_w = inner_w - GAP - left_w - 4  # sisa 4 px agar tidak muncul scrollbar horizontal
    dpg.configure_item("left_panel", width=left_w, height=body_h)
    dpg.configure_item("right_panel", width=right_w, height=body_h)
    avail_w = left_w - 2 * PAD - 4
    avail_h = max(150, body_h - CONTROLS_H)
    vw = max(160, min(avail_w, int(avail_h * ASPECT)))
    vh = int(vw / ASPECT)
    dpg.configure_item("video", width=vw, height=vh, indent=max(0, (avail_w - vw) // 2))


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def rebuild_table(metrics):
    dpg.delete_item("tbl", children_only=True, slot=1)  # hapus baris lama saja
    for m in metrics:
        k = m["key"]
        with dpg.table_row(parent="tbl"):
            dpg.add_text(m["label"], tag=f"{k}_lbl", wrap=0)
            with dpg.tooltip(f"{k}_lbl"):
                dpg.add_text(DESC.get(k, ""), wrap=320)
            dpg.add_text("-", tag=f"{k}_val")
            dpg.add_text("-", tag=f"{k}_ideal", wrap=0)
            dpg.add_progress_bar(tag=f"{k}_bar", default_value=0.0, width=-1, height=18,
                                 overlay="-")
            dpg.bind_item_theme(f"{k}_bar", "th_idle")
            dpg.add_text("-", tag=f"{k}_st")


def _label(text):
    dpg.add_text(text, color=C_MUTED)


def build_gui(app: BikeFitApp):
    dpg.create_context()
    make_status_themes()

    blank = np.zeros(app.W * app.H * 4, np.float32)
    with dpg.texture_registry():
        dpg.add_raw_texture(app.W, app.H, blank, format=dpg.mvFormat_Float_rgba, tag="tex")

    def on_file(sender, data):
        path = data.get("file_path_name")
        if path:
            app.open_source(path, True)

    with dpg.file_dialog(directory_selector=False, show=False, callback=on_file,
                         tag="filedlg", width=720, height=440, modal=True):
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            dpg.add_file_extension(ext)

    with dpg.handler_registry():  # pintasan keyboard
        dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=lambda: app.toggle_pause())
        dpg.add_key_press_handler(dpg.mvKey_R, callback=lambda: app.reset_all())

    with dpg.window(tag="main", no_scrollbar=True, no_scroll_with_mouse=True):
        # ---- header: kolom disejajarkan dengan rasio panel di bawahnya ----
        with dpg.table(tag="hdr_tbl", header_row=False, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(init_width_or_weight=LEFT_RATIO)
            dpg.add_table_column(init_width_or_weight=1.0 - LEFT_RATIO)
            with dpg.table_row():
                dpg.add_text("Bike Fitting Analyzer", tag="title_txt")
                dpg.add_text("Sumber: -", tag="src_txt", color=C_MUTED)

        with dpg.group(horizontal=True):
            # ================= PANEL KIRI: video + kontrol =================
            with dpg.child_window(tag="left_panel", border=True, width=640, height=600):
                dpg.add_image("tex", tag="video", width=640, height=360)
                dpg.add_text("", tag="status", wrap=0)
                dpg.add_separator()

                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchSame):
                    for _ in range(3):
                        dpg.add_table_column()
                    with dpg.table_row():
                        with dpg.group():
                            _label("Kamera langsung")
                            dpg.add_button(label="Webcam", tag="btn_cam", width=-1, height=30,
                                           callback=lambda: app.open_source(
                                               int(dpg.get_value("cam_idx")), False))
                        with dpg.group():
                            _label("Indeks kamera")
                            dpg.add_input_int(tag="cam_idx", default_value=0, width=-1,
                                              min_value=0, min_clamped=True)
                        with dpg.group():
                            _label("File video")
                            dpg.add_button(label="Buka Video", tag="btn_file", width=-1, height=30,
                                           callback=lambda: dpg.show_item("filedlg"))
                    with dpg.table_row():
                        dpg.add_button(label="Jeda", tag="btn_pause", width=-1, height=30,
                                       callback=lambda: app.toggle_pause())
                        dpg.add_button(label="Stop", tag="btn_stop", width=-1, height=30,
                                       callback=lambda: app.stop())
                        dpg.add_button(label="Reset Data", tag="btn_reset", width=-1, height=30,
                                       callback=lambda: app.reset_all())
                with dpg.tooltip("btn_pause"):
                    dpg.add_text("Pintasan: Spasi")
                with dpg.tooltip("btn_reset"):
                    dpg.add_text("Pintasan: R")

                dpg.add_separator()
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchSame):
                    for _ in range(3):
                        dpg.add_table_column()
                    with dpg.table_row():
                        with dpg.group():
                            _label("Sudut pandang")
                            dpg.add_combo(VIEWS, tag="view", default_value="Samping", width=-1,
                                          callback=lambda s, v: app.set_view(v))
                        with dpg.group():
                            _label("Tipe sepeda")
                            dpg.add_combo(BIKE_TYPES, tag="bike", default_value="Road", width=-1,
                                          callback=lambda: (app.evaluate(), app.update_ui()))
                        with dpg.group(tag="side_grp"):
                            _label("Sisi menghadap kamera")
                            dpg.add_radio_button(["Kanan", "Kiri"], tag="side",
                                                 default_value="Kanan", horizontal=True)
                    with dpg.table_row():
                        with dpg.group():
                            _label("Loop video")
                            dpg.add_checkbox(label="Sampai hasil pasti", tag="loop_on",
                                             default_value=True)
                        with dpg.group():
                            _label("Maks putaran")
                            dpg.add_input_int(tag="max_loops", default_value=10, width=-1,
                                              min_value=1, min_clamped=True)
                        with dpg.group():
                            _label("Putaran berjalan")
                            dpg.add_text("Hanya file video", tag="loop_info")
                dpg.add_separator()
                dpg.add_text("", tag="tips", wrap=0, color=C_MUTED)

            # ================= PANEL KANAN: skor + hasil =================
            with dpg.child_window(tag="right_panel", border=True, width=460, height=600):
                _label("SKOR BIKE FITTING")
                with dpg.table(header_row=False, policy=dpg.mvTable_SizingStretchSame):
                    dpg.add_table_column()
                    dpg.add_table_column()
                    with dpg.table_row():
                        with dpg.group(horizontal=True):
                            dpg.add_text("--", tag="overall", color=C_MUTED)
                            _label("/ 100")
                        with dpg.group():
                            dpg.add_text("Belum ada hasil", tag="overall_status", color=C_MUTED)
                            dpg.add_text("", tag="overall_ctx", color=C_MUTED)
                dpg.add_progress_bar(tag="overall_bar", default_value=0.0, width=-1, height=22,
                                     overlay="-")
                dpg.bind_item_theme("overall_bar", "th_idle")
                dpg.add_separator()

                with dpg.table(tag="tbl", header_row=True, borders_innerH=True,
                               row_background=True, policy=dpg.mvTable_SizingStretchProp):
                    for label, w in (("Parameter", 3.2), ("Nilai", 1.2), ("Ideal", 2.0),
                                     ("Skor", 2.2), ("Status", 1.5)):
                        dpg.add_table_column(label=label, init_width_or_weight=w)

                dpg.add_separator()
                _label("TITIK PERBAIKAN DAN REKOMENDASI")
                with dpg.child_window(tag="rec_panel", height=-FOOTER_H, border=True):
                    dpg.add_group(tag="rec_group")
                dpg.add_button(label="Simpan Laporan (.txt)", tag="btn_save", width=-1, height=30,
                               callback=lambda: app.save_report())
                dpg.add_text("Estimasi berbasis kamera 2D, bukan saran medis. Untuk fitting "
                             "profesional, verifikasi dengan bike fitter.",
                             tag="disclaimer", wrap=0, color=C_MUTED)

    dpg.create_viewport(title="Bike Fitting Analyzer", width=1320, height=820,
                        min_width=980, min_height=640)
    dpg.setup_dearpygui()
    dpg.set_primary_window("main", True)
    dpg.set_viewport_resize_callback(lambda s, a: layout())
    dpg.show_viewport()


def main():
    app = BikeFitApp()
    build_gui(app)
    app.set_view("Samping")
    app.show_placeholder()
    app.refresh_status()
    layout()
    while dpg.is_dearpygui_running():
        app.step()
        dpg.render_dearpygui_frame()
    app.close_source()
    dpg.destroy_context()


if __name__ == "__main__":
    main()