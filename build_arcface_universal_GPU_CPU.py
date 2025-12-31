# ================== USER SWITCH ==================
USE_GPU = True   # True = allow GPU if available, False = force CPU

# ================== ENV SETUP ==================
import os

if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================== SILENCE LOGS ==================
import time
import logging
import warnings
from contextlib import redirect_stdout
from io import StringIO

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print("🔕 Logs hidden | Building FAISS index (ArcFace | FULL DATASET)")

# ================== TF / GPU CHECK ==================
import tensorflow as tf
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices("GPU")
GPU_ACTIVE = USE_GPU and len(gpus) > 0

if GPU_ACTIVE:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("🧠 Device used : GPU")
else:
    print("🧠 Device used : CPU")

# ================== IMPORTS ==================
import numpy as np
import faiss
from tqdm import tqdm
from deepface import DeepFace

# ================== CONFIG ==================
DATASET_PATH = "dataset_6000"
MODEL_DIR = "models/arcface"

MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"

BATCH_SIZE = 48 if GPU_ACTIVE else 32

os.makedirs(MODEL_DIR, exist_ok=True)

# ================== LOAD DATA ==================
image_paths = []

for person in sorted(os.listdir(DATASET_PATH)):
    person_dir = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_dir):
        continue
    for img in os.listdir(person_dir):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append((person, os.path.join(person_dir, img)))

total_images = len(image_paths)
num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE

print(f"👤 Total identities : {len(set(p for p, _ in image_paths))}")
print(f"📸 Total images     : {total_images}")
print(f"🚀 Batch size       : {BATCH_SIZE}")
print(f"📦 Total batches    : {num_batches}\n")

# ================== EMBEDDING ==================
embeddings = []
labels = []
skipped = 0
processed = 0

start_time = time.time()

# ---------- GPU MODE (image-based ETA) ----------
if GPU_ACTIVE:
    pbar = tqdm(
        range(num_batches),
        desc="Embedding (ArcFace)",
        unit="batch",
        dynamic_ncols=True,
        mininterval=2
    )

    for batch_idx in pbar:
        batch = image_paths[
            batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE
        ]

        for person, img_path in batch:
            try:
                with redirect_stdout(StringIO()):
                    rep = DeepFace.represent(
                        img_path=img_path,
                        model_name=MODEL_NAME,
                        detector_backend=DETECTOR,
                        enforce_detection=False
                    )

                if not rep or "embedding" not in rep[0]:
                    skipped += 1
                    continue

                emb = np.array(rep[0]["embedding"], dtype="float32")
                emb /= np.linalg.norm(emb)

                embeddings.append(emb)
                labels.append(person)
                processed += 1

            except Exception:
                skipped += 1

        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0
        remaining = total_images - processed
        eta_min = (remaining / speed) / 60 if speed > 0 else 0

        pbar.set_postfix({
            "imgs": f"{processed}/{total_images}",
            "img/s": f"{speed:.2f}",
            "ETA(min)": f"{eta_min:.1f}"
        })

    pbar.close()

# ---------- CPU MODE (batch-based ETA) ----------
else:
    for batch_idx in range(num_batches):
        batch = image_paths[
            batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE
        ]

        for person, img_path in batch:
            try:
                with redirect_stdout(StringIO()):
                    rep = DeepFace.represent(
                        img_path=img_path,
                        model_name=MODEL_NAME,
                        detector_backend=DETECTOR,
                        enforce_detection=False
                    )

                if not rep or "embedding" not in rep[0]:
                    skipped += 1
                    continue

                emb = np.array(rep[0]["embedding"], dtype="float32")
                emb /= np.linalg.norm(emb)

                embeddings.append(emb)
                labels.append(person)
                processed += 1

            except Exception:
                skipped += 1

        elapsed = time.time() - start_time
        avg_batch_time = elapsed / (batch_idx + 1)
        remaining_batches = num_batches - (batch_idx + 1)
        eta_min = (remaining_batches * avg_batch_time) / 60

        print(
            f"⚡ Batch {batch_idx+1}/{num_batches} | "
            f"{avg_batch_time:.1f}s/batch | "
            f"Elapsed: {elapsed/60:.1f} min | "
            f"ETA: {eta_min:.1f} min"
        )

# ================== FINALIZE ==================
print("\n✅ Embedding generation completed")
print(f"🟢 Valid embeddings : {len(embeddings)}")
print(f"⚠️ Skipped images   : {skipped}")

embeddings = np.vstack(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(MODEL_DIR, "faiss_index.bin"))
np.save(os.path.join(MODEL_DIR, "labels.npy"), np.array(labels))

total_time = time.time() - start_time

print("\n🚀 ArcFace FAISS index built successfully")
print(f"⏱️ Total time : {total_time/60:.2f} minutes")
print(f"⚡ Avg speed  : {processed/total_time:.2f} img/s")
print(f"📦 Output    : {MODEL_DIR}")
