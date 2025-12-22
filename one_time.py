from deepface import DeepFace

DeepFace.find(
    img_path="dataset/aamir_khan/aamir.jpg",  # any one image
    db_path="dataset",
    model_name="Facenet512",
    detector_backend="retinaface",
    enforce_detection=True
)

print("✅ Embeddings built & cached")
