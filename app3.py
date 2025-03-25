import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import sqlite3

# Load YOLOv8 for object detection
model = YOLO("yolov8n.pt")  # Using the lightweight version


def detect_objects(frame):
    results = model(frame)
    return results


# Load OWL-ViT for open-world detection
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model_owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def detect_custom_objects(frame, text_queries):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(text=text_queries, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model_owl(**inputs)
    return outputs


# Database setup for storing new object embeddings
def setup_db():
    conn = sqlite3.connect("object_embeddings.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    return conn


def add_new_object(name, embedding):
    conn = setup_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO objects (name, embedding) VALUES (?, ?)", (name, embedding))
    conn.commit()
    conn.close()


def main():
    st.title("Real-Time Object & Activity Detection with Adaptive Learning")

    video_feed = st.checkbox("Enable Camera Feed")
    train_new = st.checkbox("Train New Object")

    if video_feed:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            # Detect objects
            results = detect_objects(frame)
            for r in results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    x1, y1, x2, y2 = map(int, box[:4])
                    label = model.names[int(cls)]  # Get class label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()

    if train_new:
        uploaded_image = st.file_uploader("Upload an image of the new object")
        object_name = st.text_input("Enter the object name")

        if uploaded_image and object_name:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            # Extract embeddings
            inputs = processor(images=image, return_tensors="pt")
            embedding = model_owl.get_input_embeddings()(inputs.input_ids)
            add_new_object(object_name, embedding.numpy().tobytes())
            st.success(f"Object '{object_name}' added to the database!")


if __name__ == "__main__":
    main()
