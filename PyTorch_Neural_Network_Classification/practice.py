import cv2
import numpy as np
from pathlib import Path

# ---------------------- Load Face Detector ----------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------------- Load Models ----------------------
MODEL_PATH = Path(
    "C:/Users/Haji Suleman/Desktop/Pytorch/PyTorch_Neural_Network_Classification/models"
)

# Check if models exist
for file in [
    "deploy_age.prototxt",
    "age_net.caffemodel",
    "deploy_gender.prototxt",
    "gender_net.caffemodel",
    "emotion-ferplus-8.onnx",
]:
    if not (MODEL_PATH / file).is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH / file}")

# Load networks
age_net = cv2.dnn.readNetFromCaffe(
    str(MODEL_PATH / "deploy_age.prototxt"), str(MODEL_PATH / "age_net.caffemodel")
)
gender_net = cv2.dnn.readNetFromCaffe(
    str(MODEL_PATH / "deploy_gender.prototxt"),
    str(MODEL_PATH / "gender_net.caffemodel"),
)
emotion_net = cv2.dnn.readNetFromONNX(str(MODEL_PATH / "emotion-ferplus-8.onnx"))

AGE_LIST = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]
GENDER_LIST = ["Male", "Female"]
EMOTION_LIST = [
    "Neutral",
    "Happy",
    "Surprise",
    "Sad",
    "Anger",
    "Disgust",
    "Fear",
    "Contempt",
]

# ---------------------- Start Video ----------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_img = frame[y : y + h, x : x + w].copy()

        # ---------------------- Age & Gender ----------------------
        blob = cv2.dnn.blobFromImage(
            face_img,
            1.0,
            (227, 227),
            [78.4263377603, 87.7689143744, 114.895847746],
            swapRB=False,
        )
        # Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]

        # ---------------------- Expression ----------------------
        face_gray = cv2.resize(gray[y : y + h, x : x + w], (64, 64))
        face_blob = cv2.dnn.blobFromImage(
            face_gray, 1 / 255.0, (64, 64), [0, 0, 0], swapRB=True, crop=False
        )
        emotion_net.setInput(face_blob)
        emotion_preds = emotion_net.forward()
        emotion = EMOTION_LIST[emotion_preds[0].argmax()]

        # ---------------------- Draw & Glow ----------------------
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Cyber glow effect
        glow = frame.copy()
        cv2.rectangle(glow, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 255), 6)
        frame = cv2.addWeighted(glow, 0.3, frame, 0.7, 0)

        # Put Age, Gender, Expression
        cv2.putText(
            frame,
            f"{gender}, {age}",
            (x, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Expression: {emotion}",
            (x, y - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    cv2.imshow("Face AI System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
