from deepface import DeepFace
import cv2
import os
import numpy as np

# ---------------- CONFIG ----------------
DATABASE_PATH = os.path.abspath("dataset")   # dataset folder with .pkl inside
MODEL_NAME = "Facenet512"
DETECTOR = "retinaface"
THRESHOLD = 0.3                              # strict threshold
FRAME_SKIP = 15                              # process every Nth frame
CAMERA_INDEX = 0                             # 0 = default webcam

# ---------------- INIT ----------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam connected")

frame_count = 0
last_identity = "Unknown"
last_confidence = 0.0
last_box = None
last_distance = None

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    if frame_count % FRAME_SKIP == 0:
        try:
            detections = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR,
                enforce_detection=False
            )

            if detections:
                face = detections[0]
                area = face["facial_area"]

                x, y, w, h = area["x"], area["y"], area["w"], area["h"]

                # -------- SAFE CROP --------
                h_frame, w_frame, _ = frame.shape
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_frame, x + w)
                y2 = min(h_frame, y + h)

                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    last_identity = "Unknown"
                    last_box = None
                else:
                    results = DeepFace.find(
                        img_path=face_img,
                        db_path=DATABASE_PATH,
                        model_name=MODEL_NAME,
                        detector_backend=DETECTOR,
                        enforce_detection=False,
                        refresh_database=False   # 🔥 THIS IS THE FIX
                    )

                    if len(results) > 0 and not results[0].empty:
                        best = results[0].iloc[0]
                        distance = float(best["distance"])
                        last_distance = distance

                        if distance <= THRESHOLD:
                            identity_path = best["identity"]
                            last_identity = os.path.basename(
                                os.path.dirname(identity_path)
                            )
                            last_confidence = max(
                                0.0,
                                min(100.0, 100 * (1 - distance / THRESHOLD))
                            )
                            last_box = (x1, y1, x2 - x1, y2 - y1)
                        else:
                            last_identity = "Unknown"
                            last_confidence = 0.0
                            last_box = None
                    else:
                        last_identity = "Unknown"
                        last_confidence = 0.0
                        last_box = None

        except Exception as e:
            print("⚠️ Error:", e)
            last_identity = "Unknown"
            last_box = None

    # ---------------- DRAW ----------------
    if last_box:
        x, y, w, h = last_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{last_identity} ({last_confidence:.1f}%)"
        color = (0, 255, 0)
    else:
        label = "Unknown"
        color = (0, 0, 255)

    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    if last_distance is not None:
        cv2.putText(
            frame,
            f"Distance: {last_distance:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    cv2.imshow("DeepFace Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam stopped")
