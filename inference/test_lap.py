# ── test_laptop_cam.py ────────────────────────────────────────────────────────
import cv2
from ultralytics import YOLO

model = YOLO("C:/YOLOv8/best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot access laptop camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
ret, test_frame = cap.read()
h, w = test_frame.shape[:2]
out = cv2.VideoWriter("C:/YOLOv8/laptop_cam_output_10.mp4", fourcc, 20, (w, h))

print("Laptop camera running — press 'q' to stop and save.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model.predict(
        source=frame,
        imgsz=416,        # matches training resolution
        conf=0.55,
        iou=0.45,
        verbose=False
    )

    annotated = results[0].plot()

    n_det = len(results[0].boxes)
    label = f"[LAPTOP CAM] Detected: {n_det} victim(s)"
    color = (0, 200, 0) if n_det == 0 else (0, 0, 255)
    cv2.putText(annotated, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    for box in results[0].boxes:
        conf = float(box.conf[0])
        x1, y1 = map(int, box.xyxy[0][:2])
        cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    out.write(annotated)
    cv2.imshow("Victim Detection — Laptop Camera", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved to: C:/YOLOv8/laptop_cam_output_10.mp4")