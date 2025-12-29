# ================== SILENCE LOGS ==================
import os
import time
import logging
import warnings
from contextlib import redirect_stdout
from io import StringIO

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print("🔕 Logs hidden | FAISS Test recognition (ArcFace)")

# ================== IMPORTS ==================
import cv2
import numpy as np
import faiss
from deepface import DeepFace

# ================== CONFIG ==================
MODEL_DIR = "models/arcface"              # 👈 ArcFace models folder
TEST_IMAGE = "test_images/weeknd_test.png"  # image NOT in dataset

MODEL_NAME = "ArcFace"
DETECTOR = "mtcnn"

THRESHOLD = 0.8          # 🔥 ArcFace normalized L2 threshold
RESIZE_TO = (640, 480)   # 🔥 speed boost

# ================== LOAD FAISS ==================
index = faiss.read_index(os.path.join(MODEL_DIR, "faiss_index.bin"))
labels = np.load(os.path.join(MODEL_DIR, "labels.npy"))

print("✅ FAISS index loaded")
print("🔍 Running test recognition...\n")

total_start = time.time()

# ================== LOAD + RESIZE IMAGE ==================
img = cv2.imread(TEST_IMAGE)
if img is None:
    raise ValueError("❌ Failed to load test image")

img = cv2.resize(img, RESIZE_TO)

# ================== FACE DETECTION ==================
detect_start = time.time()
with redirect_stdout(StringIO()):
    faces = DeepFace.extract_faces(
        img_path=img,
        detector_backend=DETECTOR,
        enforce_detection=True
    )
detect_time = time.time() - detect_start

print(f"👁️ Face detection time : {detect_time:.2f} sec")

if not faces:
    print("\n❌ No face detected")
    exit()

face_img = faces[0]["face"]

# ================== EMBEDDING ==================
embed_start = time.time()
with redirect_stdout(StringIO()):
    rep = DeepFace.represent(
        img_path=face_img,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False
    )

embedding = np.array(rep[0]["embedding"], dtype="float32")
embedding /= np.linalg.norm(embedding)
embedding = embedding.reshape(1, -1)

embed_time = time.time() - embed_start

# ================== FAISS SEARCH ==================
search_start = time.time()
D, I = index.search(embedding, k=1)
search_time = time.time() - search_start

distance = float(D[0][0])
idx = int(I[0][0])
predicted_name = labels[idx]

total_time = time.time() - total_start

# ================== RESULT ==================
print("\n🔎 FAISS nearest match")
print(f"👤 Nearest identity   : {predicted_name}")
print(f"📏 Distance           : {distance:.3f}")

if distance <= THRESHOLD:
    print(f"\n🎯 FINAL RESULT       : RECOGNIZED AS  →  {predicted_name}")
else:
    print("\n❌ FINAL RESULT       : UNKNOWN (rejected by threshold)")

print(f"\n👁️ Face detection time : {detect_time:.2f} sec")
print(f"🧠 Embedding time     : {embed_time:.2f} sec")
print(f"⚡ FAISS search time  : {search_time:.4f} sec")
print(f"⏱️ Total time         : {total_time:.2f} sec")
