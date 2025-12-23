# ================== SILENCE LOGS ==================
import os
import logging
import warnings
from contextlib import redirect_stdout
from io import StringIO

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print("🔕 Logs hidden | FAISS Live recognition")

# ================== IMPORTS ==================
import time
import cv2
import numpy as np
import faiss
import threading
from deepface import DeepFace

# ================== CONFIG ==================
DATASET_PATH = "dataset"
MODEL_NAME = "Facenet512"
DETECTOR = "retinaface"

THRESHOLD = 1.0          # FAISS L2 distance (normalized embeddings)
FRAME_SKIP = 15
CAMERA_INDEX = 0

# ================== LOAD FAISS ==================
index = faiss.read_index(os.path.join(DATASET_PATH, "faiss_index.bin"))
labels = np.load(os.path.join(DATASET_PATH, "labels.npy"))

print("✅ FAISS index loaded")
print("🎥 Webcam started | Live feed running")

# ================== SHARED STATE ==================
last_identity = "Unknown"
last_distance = None
last_confidence = 0.0
last_box = None
processing = False

start_time = time.time()
first_match_time = None

# ================== BACKGROUND RECOGNITION ==================
def recognize(frame):
    global last_identity, last_distance, last_confidence
    global last_box, processing, first_match_time

    try:
        # ---------- FACE DETECTION ----------
        t_detect_start = time.time()
        with redirect_stdout(StringIO()):
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTOR,
                enforce_detection=False
            )
        detect_time = time.time() - t_detect_start

        if not faces:
            processing = False
            return

        area = faces[0]["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]

        h_f, w_f, _ = frame.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_f, x + w), min(h_f, y + h)
        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            processing = False
            return

        # ---------- EMBEDDING ----------
        t_embed_start = time.time()
        with redirect_stdout(StringIO()):
            rep = DeepFace.represent(
                img_path=face_img,
                model_name=MODEL_NAME,
                detector_backend="skip",   # 🔥 face already detected
                enforce_detection=False
            )
        embed_time = time.time() - t_embed_start

        embedding = np.array(rep[0]["embedding"], dtype="float32")
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.reshape(1, -1)

        # ---------- FAISS SEARCH ----------
        t_search_start = time.time()
        D, I = index.search(embedding, k=1)
        search_time = time.time() - t_search_start

        distance = float(D[0][0])
        idx = int(I[0][0])

        if distance <= THRESHOLD:
            last_identity = labels[idx]
            last_distance = distance
            last_confidence = 100 * (1 - distance / THRESHOLD)
            last_box = (x1, y1, x2 - x1, y2 - y1)

            if first_match_time is None:
                first_match_time = time.time()
                print("\n🎯 First successful recognition")
                print(f"👁️ Face detection time : {detect_time:.2f} sec")
                print(f"🧠 Embedding time      : {embed_time:.2f} sec")
                print(f"⚡ FAISS search time   : {search_time:.4f} sec")
                print(
                    f"⏱️ Total time          : "
                    f"{first_match_time - start_time:.2f} sec\n"
                )
        else:
            last_identity = "Unknown"
            last_box = None

    except Exception:
        pass

    processing = False


# ================== INIT CAMERA ==================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam")

frame_count = 0

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # ---------- BACKGROUND FAISS RECOGNITION ----------
    if frame_count % FRAME_SKIP == 0 and not processing:
        processing = True
        threading.Thread(
            target=recognize,
            args=(frame.copy(),),
            daemon=True
        ).start()

    # ---------- UI ----------
    if last_box:
        x, y, w, h = last_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{last_identity} ({last_confidence:.1f}%)"
        color = (0, 255, 0)
    else:
        label = "Unknown"
        color = (0, 0, 255)

    cv2.putText(
        frame, label, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
    )

    if last_distance is not None:
        cv2.putText(
            frame, f"Distance: {last_distance:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 0), 2
        )

    cv2.imshow("FAISS Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam stopped")
