"""
NeuroMood - Enhanced Mood Detection
====================================
Built on top of RaiyanRahman/Mood-Detection core logic.

Enhancements added:
  - Emoji overlay per detected emotion
  - Confidence bar for each label
  - Dominant emotion summary (last 5 seconds)
  - Real-time emotion history graph (last 30 seconds)
  - CSV mood logger with timestamps
  - Multi-face tracking (each face labeled separately)
  - Sound feedback on emotion change
  - Anger/Fear alert with desktop notification after 3s

Requirements:
  pip install opencv-python tensorflow keras numpy pillow plyer deepface
"""

import cv2
import numpy as np
import time
import csv
import os
import threading
import collections
from datetime import datetime
from deepface import DeepFace

# ── Optional imports (graceful fallback if not installed) ──────────────────────
try:
    from plyer import notification as plyer_notif
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    print("[WARN] plyer not installed — desktop notifications disabled. Run: pip install plyer")

try:
    import winsound
    SOUND_BACKEND = "winsound"
except ImportError:
    try:
        import subprocess
        SOUND_BACKEND = "afplay"      # macOS
    except Exception:
        SOUND_BACKEND = None
        print("[WARN] No sound backend found — sound feedback disabled.")

# ── Config ─────────────────────────────────────────────────────────────────────
CASCADE_PATH  = "haarcascade_frontalface_default.xml"
LOG_PATH      = "mood_log.csv"
GRAPH_SECONDS = 30          # seconds of history shown in graph
DOMINANT_WINDOW = 5         # seconds for dominant emotion summary
ALERT_SECONDS = 3           # seconds before anger/fear triggers notification
GRAPH_H, GRAPH_W = 160, 400 # graph panel dimensions

# ── Emotion config ─────────────────────────────────────────────────────────────
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

EMOJIS = {
    'angry':    '😡',
    'disgust':  '🤢',
    'fear':     '😨',
    'happy':    '😊',
    'sad':      '😢',
    'surprise': '😲',
    'neutral':  '😐',
}

# BGR colors per emotion for the graph lines / boxes
COLORS = {
    'angry':    (60,  60,  220),
    'disgust':  (60,  180, 60),
    'fear':     (180, 60,  180),
    'happy':    (60,  210, 255),
    'sad':      (220, 120, 60),
    'surprise': (60,  220, 200),
    'neutral':  (180, 180, 180),
}

ALERT_EMOTIONS = {'angry', 'fear'}

# ── CSV logger ─────────────────────────────────────────────────────────────────
def init_csv():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "face_id", "emotion", "confidence"])

def log_emotion(face_id, emotion, confidence):
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         face_id, emotion, f"{confidence:.3f}"])

# ── Sound feedback ─────────────────────────────────────────────────────────────
_last_sound_emotion = {}  # face_id -> last emotion that triggered sound
_sound_lock = threading.Lock()

SOUND_FREQS = {
    'happy':    880,
    'surprise': 660,
    'sad':      330,
    'angry':    200,
    'fear':     250,
    'disgust':  300,
    'neutral':  440,
}

def play_sound_async(emotion):
    def _play():
        try:
            if SOUND_BACKEND == "winsound":
                winsound.Beep(SOUND_FREQS.get(emotion, 440), 150)
            elif SOUND_BACKEND == "afplay":
                freq = SOUND_FREQS.get(emotion, 440)
                subprocess.run(
                    ["sox", "-n", "-d", "synth", "0.15", "sine", str(freq)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
        except Exception:
            pass
    if SOUND_BACKEND:
        threading.Thread(target=_play, daemon=True).start()

def maybe_play_sound(face_id, emotion):
    with _sound_lock:
        if _last_sound_emotion.get(face_id) != emotion:
            _last_sound_emotion[face_id] = emotion
            play_sound_async(emotion)

# ── Desktop notification ───────────────────────────────────────────────────────
_notif_sent_at = 0
_notif_cooldown = 15  # seconds between notifications

def maybe_notify(emotion):
    global _notif_sent_at
    now = time.time()
    if not PLYER_AVAILABLE:
        return
    if now - _notif_sent_at < _notif_cooldown:
        return
    _notif_sent_at = now
    threading.Thread(
        target=plyer_notif.notify,
        kwargs=dict(
            title="⚠️ NeuroMood Alert",
            message=f"Sustained {emotion.upper()} detected for {ALERT_SECONDS}+ seconds. Take a breath! 🧘",
            timeout=5,
        ),
        daemon=True
    ).start()

# ── Dominant emotion tracker ───────────────────────────────────────────────────
class EmotionHistory:
    """Tracks timestamped emotions for one face."""
    def __init__(self):
        self.records = collections.deque()          # (timestamp, emotion, conf)
        self.alert_start = None                     # when ALERT_EMOTIONS started

    def push(self, emotion, confidence):
        now = time.time()
        self.records.append((now, emotion, confidence))
        # prune old entries beyond GRAPH_SECONDS
        while self.records and now - self.records[0][0] > GRAPH_SECONDS:
            self.records.popleft()

    def dominant(self, window=DOMINANT_WINDOW):
        now = time.time()
        recent = [r[1] for r in self.records if now - r[0] <= window]
        if not recent:
            return None
        return max(set(recent), key=recent.count)

    def check_alert(self, emotion):
        """Returns True if alert emotion sustained for ALERT_SECONDS."""
        now = time.time()
        if emotion in ALERT_EMOTIONS:
            if self.alert_start is None:
                self.alert_start = now
            elif now - self.alert_start >= ALERT_SECONDS:
                return True
        else:
            self.alert_start = None
        return False

    def graph_series(self):
        """Returns dict emotion -> list of (t_relative, conf) for graphing."""
        now = time.time()
        series = {e: [] for e in EMOTIONS}
        for ts, em, cf in self.records:
            series[em].append((now - ts, cf))
        return series

# ── Draw helpers ───────────────────────────────────────────────────────────────

def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=8, thickness=-1, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_confidence_bar(frame, x, y, w, label, confidence, color):
    bar_w = int(w * confidence)
    bar_h = 12
    cv2.rectangle(frame, (x, y), (x + w, y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), color, -1)
    cv2.putText(frame, f"{label} {confidence*100:.0f}%",
                (x + w + 6, y + bar_h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

def put_emoji_text(frame, emoji, x, y, size=1.4):
    """Render emoji as OpenCV text fallback (ASCII representation)."""
    ascii_map = {
        '😡': '>:(', '🤢': 'eww', '😨': 'D:',
        '😊': ':)',  '😢': ':(',  '😲': ':O', '😐': ':|'
    }
    text = ascii_map.get(emoji, '?')
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                size, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                size, (80, 80, 80), 1, cv2.LINE_AA)

def draw_emotion_graph(histories):
    """Draw a mini graph panel showing emotion confidence over last 30s."""
    panel = np.zeros((GRAPH_H, GRAPH_W, 3), dtype=np.uint8)
    panel[:] = (25, 25, 35)

    cv2.putText(panel, "Emotion History (30s)", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 200), 1, cv2.LINE_AA)

    if not histories:
        return panel

    history = next(iter(histories.values()))
    series  = history.graph_series()

    plot_x0, plot_y0 = 8, 24
    plot_w = GRAPH_W - 16
    plot_h = GRAPH_H - 32

    for i in range(5):
        gy = plot_y0 + int(plot_h * i / 4)
        cv2.line(panel, (plot_x0, gy), (plot_x0 + plot_w, gy), (50, 50, 60), 1)

    for emotion in EMOTIONS:
        pts = series[emotion]
        if len(pts) < 2:
            continue
        color = COLORS[emotion]
        poly  = []
        for t_ago, conf in pts:
            px = plot_x0 + plot_w - int((t_ago / GRAPH_SECONDS) * plot_w)
            py = plot_y0 + plot_h - int(conf * plot_h)
            px = max(plot_x0, min(plot_x0 + plot_w, px))
            py = max(plot_y0, min(plot_y0 + plot_h, py))
            poly.append((px, py))
        for i in range(1, len(poly)):
            cv2.line(panel, poly[i-1], poly[i], color, 1, cv2.LINE_AA)

    lx = 8
    for i, em in enumerate(EMOTIONS):
        col = COLORS[em]
        cv2.rectangle(panel, (lx, GRAPH_H - 10), (lx + 10, GRAPH_H - 2), col, -1)
        cv2.putText(panel, em[:3], (lx + 13, GRAPH_H - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)
        lx += 52

    return panel

def draw_dominant_summary(frame, histories):
    """Top-right corner: dominant emotion over last 5s."""
    fh, fw = frame.shape[:2]
    lines = []
    for fid, hist in histories.items():
        dom = hist.dominant()
        if dom:
            lines.append((fid, dom))

    if not lines:
        return

    box_x = fw - 220
    box_y = 10
    draw_rounded_rect(frame, box_x - 4, box_y - 4, fw - 8, box_y + len(lines) * 26 + 10,
                      (30, 30, 30), radius=6, alpha=0.5)

    cv2.putText(frame, "Dominant (5s)", (box_x, box_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)

    for i, (fid, dom) in enumerate(lines):
        col = COLORS[dom]
        label = f"Face {fid+1}: {dom.upper()}"
        cv2.putText(frame, label, (box_x, box_y + 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_csv()

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    histories = {}   # face_id (int) -> EmotionHistory

    print("[INFO] NeuroMood running. Press Q to quit.")
    print(f"[INFO] Logging emotions to: {LOG_PATH}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        active_ids = set()

        for face_id, (fx, fy, fw_f, fh_f) in enumerate(faces):
            active_ids.add(face_id)
            if face_id not in histories:
                histories[face_id] = EmotionHistory()

            # ── DeepFace prediction ───────────────────────────────────────────
            try:
                face_crop = frame[fy:fy + fh_f, fx:fx + fw_f]
                result = DeepFace.analyze(face_crop, actions=['emotion'],
                                          enforce_detection=False, silent=True)
                emotion_scores = result[0]['emotion']
                emotion = max(emotion_scores, key=emotion_scores.get).lower()
                confidence = float(emotion_scores[emotion]) / 100.0
                preds = np.array([emotion_scores.get(e, 0) / 100.0 for e in EMOTIONS])
            except Exception:
                emotion = 'neutral'
                confidence = 1.0
                preds = np.zeros(len(EMOTIONS))

            color = COLORS.get(emotion, (180, 180, 180))

            # ── Update history & side effects ─────────────────────────────────
            histories[face_id].push(emotion, confidence)
            log_emotion(face_id, emotion, confidence)
            maybe_play_sound(face_id, emotion)

            if histories[face_id].check_alert(emotion):
                maybe_notify(emotion)

            # ── Draw face box ─────────────────────────────────────────────────
            cv2.rectangle(frame, (fx, fy), (fx + fw_f, fy + fh_f), color, 2)

            # Face ID badge
            badge = f"Face {face_id + 1}"
            cv2.rectangle(frame, (fx, fy - 22), (fx + 65, fy), (40, 40, 40), -1)
            cv2.putText(frame, badge, (fx + 3, fy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # ── Emoji ─────────────────────────────────────────────────────────
            emoji = EMOJIS.get(emotion, '?')
            put_emoji_text(frame, emoji, fx + fw_f + 6, fy + 32, size=1.2)

            # ── Confidence bars (all 7 emotions) ──────────────────────────────
            bar_x  = fx
            bar_y0 = fy + fh_f + 6
            bar_w  = min(fw_f, 120)

            draw_rounded_rect(frame,
                              bar_x - 2, bar_y0 - 2,
                              bar_x + bar_w + 110, bar_y0 + len(EMOTIONS) * 16 + 4,
                              (20, 20, 20), radius=4, alpha=0.5)

            for j, em in enumerate(EMOTIONS):
                draw_confidence_bar(frame,
                                    bar_x, bar_y0 + j * 16,
                                    bar_w, em, float(preds[j]),
                                    COLORS[em])

        # Clean up histories for faces no longer detected
        for old_id in list(histories.keys()):
            if old_id not in active_ids:
                del histories[old_id]

        # ── Dominant emotion summary (top-right) ──────────────────────────────
        draw_dominant_summary(frame, histories)

        # ── Emotion history graph (bottom of frame) ───────────────────────────
        graph = draw_emotion_graph(histories)
        gx = (fw - GRAPH_W) // 2
        gy = fh - GRAPH_H - 8
        if gy > 0 and gx >= 0:
            roi_g = frame[gy:gy + GRAPH_H, gx:gx + GRAPH_W]
            blended = cv2.addWeighted(roi_g, 0.3, graph, 0.7, 0)
            frame[gy:gy + GRAPH_H, gx:gx + GRAPH_W] = blended

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.putText(frame, f"NeuroMood  |  Faces: {len(faces)}  |  Q to quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 220), 1, cv2.LINE_AA)

        cv2.imshow("NeuroMood", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Session ended. Mood log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()