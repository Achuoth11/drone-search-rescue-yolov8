import cv2
import json
import time
import os
import socket
import struct
import threading
import subprocess
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════

MODEL_PATH = "/home/pi/wheelchair_project/best.pt"

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FPS_TARGET   = 30

CONF_THRESH  = 0.55
INPUT_SIZE   = 640    # ← faster on Pi (trained at 960, runtime at 320)

STREAM_ENABLED  = True
STREAM_PORT     = 8554
STREAM_QUALITY  = 70

LOG_FILE = Path("/home/pi/wheelchair_project/detections.json")

SAVE_VIDEO       = True
VIDEO_OUTPUT_DIR = Path("/home/pi/wheelchair_project/videos")
VIDEO_CODEC      = "mp4v"
VIDEO_FPS_SAVE   = 15

CLASS_NAMES = {0: "Person"}

# ── CHANGE 1: Force live window ON always ─────────────
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')  # fix Qt warning on Pi
SHOW_WINDOW = True                                 # always show on Pi screen

# ── Camera Mode ───────────────────────────────────────
CAMERA_MODE = "PI"    # "PI" = rpicam-vid | "USB" = webcam


# ═══════════════════════════════════════════════════════
#  PI CAMERA HANDLER (rpicam-vid — no picamera2 needed)
# ═══════════════════════════════════════════════════════

class PiCameraHandler:
    def __init__(self, width=FRAME_WIDTH, height=FRAME_HEIGHT,
                 fps=FPS_TARGET):
        self.width  = width
        self.height = height
        self.fps    = fps
        self.pipe   = None
        self.buf    = b""
        self._open()

    def _open(self):
        print("[CAM] Starting Pi Camera (rpicam-vid)...")
        for cmd_name in ["rpicam-vid", "libcamera-vid"]:
            cmd = [
                cmd_name,
                "--width",     str(self.width),
                "--height",    str(self.height),
                "--framerate", str(self.fps),
                "--codec",     "mjpeg",
                "--output",    "-",
                "--nopreview",
                "-t",          "0",
            ]
            try:
                self.pipe = subprocess.Popen(
                    cmd,
                    stdout  = subprocess.PIPE,
                    stderr  = subprocess.DEVNULL,
                    bufsize = 0
                )
                time.sleep(2.0)
                if self.pipe.poll() is None:
                    print(f"[CAM] ✅ Pi Camera via {cmd_name} — "
                          f"{self.width}x{self.height} @ {self.fps}fps")
                    return
                self.pipe = None
            except FileNotFoundError:
                self.pipe = None
                continue
        print("[ERROR] rpicam-vid not found! Check ribbon cable.")
        exit(1)

    def read(self):
        try:
            while True:
                chunk = self.pipe.stdout.read(4096)
                if not chunk:
                    return False, None
                self.buf += chunk
                start = self.buf.find(b'\xff\xd8')
                end   = self.buf.find(b'\xff\xd9')
                if start != -1 and end != -1 and end > start:
                    jpg      = self.buf[start:end + 2]
                    self.buf = self.buf[end + 2:]
                    frame    = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
        except Exception as e:
            print(f"[WARN] Camera read error: {e}")
            return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:  return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return float(self.height)
        if prop == cv2.CAP_PROP_FPS:          return float(self.fps)
        return 0.0

    def isOpened(self):
        return self.pipe is not None and self.pipe.poll() is None

    def release(self):
        if self.pipe:
            self.pipe.terminate()
            self.pipe.wait()
            print("[CAM] Pi Camera released")


# ═══════════════════════════════════════════════════════
#  OPEN CAMERA
# ═══════════════════════════════════════════════════════

def open_camera():
    if CAMERA_MODE == "PI":
        print("[CAM] Mode: PI CAMERA (rpicam-vid)")
        return PiCameraHandler(FRAME_WIDTH, FRAME_HEIGHT, FPS_TARGET)
    else:
        print("[CAM] Mode: USB WEBCAM")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          FPS_TARGET)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            print("[ERROR] Cannot open USB webcam")
            exit(1)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAM] ✅ USB Webcam — {w}x{h}")
        return cap


# ═══════════════════════════════════════════════════════
#  LOAD YOLO MODEL
# ═══════════════════════════════════════════════════════

def load_model():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        exit(1)
    print(f"[MODEL] Loading: {MODEL_PATH}")
    model = YOLO(str(model_path))
    print("[MODEL] ✅ YOLO model loaded")
    return model


# ═══════════════════════════════════════════════════════
#  RUN INFERENCE
# ═══════════════════════════════════════════════════════

def run_inference(model, frame):
    t0 = time.time()
    results = model.predict(
        source  = frame,
        imgsz   = INPUT_SIZE,
        conf    = CONF_THRESH,
        verbose = False
    )
    infer_ms = (time.time() - t0) * 1000
    return results, infer_ms


# ═══════════════════════════════════════════════════════
#  PARSE DETECTIONS
# ═══════════════════════════════════════════════════════

def parse_detections(results):
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id        = int(box.cls[0])
            conf          = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id":   cls_id,
                "class_name": CLASS_NAMES.get(cls_id, "Person"),
                "confidence": round(conf, 3),
                "bbox":       [x1, y1, x2, y2]
            })
    return detections


# ═══════════════════════════════════════════════════════
#  DRAW DETECTIONS
# ═══════════════════════════════════════════════════════

def draw_detections(frame, detections, fps, infer_ms, frame_no):
    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf  = det["confidence"]
        label = f"{det['class_name']} {conf:.0%}"
        color = (0, int(100 + 155 * conf), 50)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, label_y),
                      (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

    overlay = [
        f"FPS: {fps:.1f}  |  Infer: {infer_ms:.0f}ms  |  imgsz={INPUT_SIZE}",
        f"Persons: {len(detections)}  |  Frame: {frame_no}",
        f"Model: best.pt  |  conf={CONF_THRESH}",
    ]
    for i, text in enumerate(overlay):
        y = 24 + i * 24
        cv2.putText(frame, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    n = len(detections)
    if n > 0:
        banner = f"  VICTIM DETECTED: {n}  "
        (bw, bh), _ = cv2.getTextSize(
            banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        bx = (w - bw) // 2
        cv2.rectangle(frame, (bx - 6, 8),
                      (bx + bw + 6, 8 + bh + 12), (0, 0, 200), -1)
        cv2.putText(frame, banner, (bx, 8 + bh + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, h - 28), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "Raspberry Pi SAR — LIVE WINDOW",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1)
    return frame


# ═══════════════════════════════════════════════════════
#  STREAM SERVER
# ═══════════════════════════════════════════════════════

class StreamServer:
    def __init__(self, port=STREAM_PORT):
        self.port    = port
        self.clients = []
        self.lock    = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print(f"[STREAM] TCP server on port {self.port}")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            pi_ip = s.getsockname()[0]
            s.close()
            print(f"\n[STREAM] ✅ On laptop set:")
            print(f"           STREAM_URL = 'tcp://{pi_ip}:{self.port}'\n")
        except Exception:
            print(f"[STREAM] STREAM_URL = 'tcp://PI_IP:{self.port}'")

    def _accept_loop(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(5)
        while self.running:
            try:
                conn, addr = server.accept()
                print(f"[STREAM] Laptop connected: {addr[0]}")
                with self.lock:
                    self.clients.append(conn)
            except Exception:
                break

    def send_frame(self, frame):
        if not self.clients:
            return
        _, jpeg = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        data   = jpeg.tobytes()
        header = struct.pack(">L", len(data))
        dead   = []
        with self.lock:
            for client in self.clients:
                try:
                    client.sendall(header + data)
                except Exception:
                    dead.append(client)
            for d in dead:
                self.clients.remove(d)

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════
#  VIDEO WRITER
# ═══════════════════════════════════════════════════════

class VideoWriter:
    def __init__(self, width, height):
        VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_path = VIDEO_OUTPUT_DIR / f"rescue_{ts}.mp4"
        fourcc        = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self.writer   = cv2.VideoWriter(
            str(self.out_path), fourcc,
            VIDEO_FPS_SAVE, (width, height))
        print(f"[VIDEO] 🎥 Recording: {self.out_path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
        print(f"[VIDEO] ✅ Saved: {self.out_path}")


# ═══════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════

detection_log = []

def log_detection(timestamp, detections, fps, infer_ms):
    if detections:
        detection_log.append({
            "timestamp":  timestamp,
            "count":      len(detections),
            "fps":        round(fps, 1),
            "infer_ms":   round(infer_ms, 1),
            "detections": detections,
        })

def save_log():
    if detection_log:
        with open(LOG_FILE, "w") as f:
            json.dump(detection_log, f, indent=2)
        print(f"[LOG] Saved {len(detection_log)} events → {LOG_FILE}")


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("  Raspberry Pi — YOLOv8 Human Detection (LIVE)")
    print(f"  Model   : best.pt")
    print(f"  Camera  : {CAMERA_MODE}")
    print(f"  ImgSz   : {INPUT_SIZE} (trained=960, runtime={INPUT_SIZE})")
    print(f"  Conf    : {CONF_THRESH}")
    print(f"  Window  : LIVE on Pi screen — press q to quit")
    print("=" * 58 + "\n")

    model = load_model()
    cap   = open_camera()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Warm up camera
    print("[CAM] Warming up...")
    for _ in range(5):
        cap.read()
    print("[CAM] ✅ Ready\n")

    # Stream server
    streamer = None
    if STREAM_ENABLED:
        streamer = StreamServer()
        streamer.start()

    # Video writer
    video_writer = None
    if SAVE_VIDEO:
        video_writer = VideoWriter(actual_w, actual_h)

    # ── CHANGE 2: Open live window on Pi screen ────────
    cv2.namedWindow("Pi SAR Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pi SAR Detection", actual_w, actual_h)
    print("[WINDOW] ✅ Live window open on Pi screen — press q to quit\n")

    frame_count   = 0
    total_persons = 0
    start_time    = time.time()
    rolling_times = []

    print(f"  {'Frame':<8} {'Persons':<9} {'FPS':<8} {'InferMs':<10} Status")
    print(f"  {'─'*52}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            results, infer_ms = run_inference(model, frame)
            detections        = parse_detections(results)

            rolling_times.append(time.time())
            if len(rolling_times) > 15:
                rolling_times.pop(0)
            fps = ((len(rolling_times) - 1)
                   / (rolling_times[-1] - rolling_times[0])
                   if len(rolling_times) >= 2 else 0.0)

            total_persons += len(detections)
            status = (f"★ DETECTED ({len(detections)})"
                      if detections else "scanning...")
            print(f"  {frame_count:<8} {len(detections):<9} "
                  f"{fps:<7.1f}  {infer_ms:<9.1f}  {status}")

            for det in detections:
                if det["confidence"] >= 0.60:
                    print(f"           *** {det['class_name']} "
                          f"{det['confidence']:.0%} @ {det['bbox']}")

            log_detection(
                datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                detections, fps, infer_ms)

            annotated = draw_detections(
                frame.copy(), detections, fps, infer_ms, frame_count)

            # ── CHANGE 3: Always show window on Pi ────────
            cv2.imshow("Pi SAR Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Quit key pressed")
                break

            if streamer:
                streamer.send_frame(annotated)
            if video_writer:
                video_writer.write(annotated)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by Ctrl+C")

    finally:
        elapsed = time.time() - start_time
        cap.release()
        cv2.destroyAllWindows()
        if video_writer:
            video_writer.release()
        if streamer:
            streamer.stop()
        save_log()
        print(f"\n{'═'*58}")
        print(f"  Frames : {frame_count} | Detections: {total_persons}")
        print(f"  Avg FPS: {frame_count/max(elapsed,1):.1f}")
        print(f"  Log    : {LOG_FILE}")
        print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()