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

print("🔕 Logs hidden | FAISS Test recognition")

# ================== IMPORTS ==================
import numpy as np
import faiss
from deepface import DeepFace

# ================== CONFIG ==================
DATASET_PATH = "dataset"
TEST_IMAGE = "amir_test.png"   # image NOT present in dataset
MODEL_NAME = "Facenet512"
DETECTOR = "retinaface"
THRESHOLD = 1.0               # normalized FaceNet512 + L2

# ================== LOAD FAISS ==================
index = faiss.read_index(os.path.join(DATASET_PATH, "faiss_index.bin"))
labels = np.load(os.path.join(DATASET_PATH, "labels.npy"))

print("✅ FAISS index loaded")
print("🔍 Running test recognition...\n")

total_start = time.time()

# ================== FACE DETECTION ==================
detect_start = time.time()
with redirect_stdout(StringIO()):
    faces = DeepFace.extract_faces(
        img_path=TEST_IMAGE,
        detector_backend=DETECTOR,
        enforce_detection=True
    )
detect_time = time.time() - detect_start

print(f"👁️ Face detection time : {detect_time:.2f} sec")

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
embedding = embedding / np.linalg.norm(embedding)
embedding = embedding.reshape(1, -1)
embed_time = time.time() - embed_start

# ================== FAISS SEARCH ==================
search_start = time.time()
D, I = index.search(embedding, k=1)
search_time = time.time() - search_start

distance = float(D[0][0])
idx = int(I[0][0])

total_time = time.time() - total_start

# ================== RESULT ==================
if distance <= THRESHOLD:
    print(f"✅ Recognized as: {labels[idx]}")
else:
    print("❌ Unknown face")

print(f"📏 Distance           : {distance:.3f}")
print(f"🧠 Embedding time     : {embed_time:.2f} sec")
print(f"⚡ FAISS search time  : {search_time:.4f} sec")
print(f"⏱️ Total time         : {total_time:.2f} sec")
