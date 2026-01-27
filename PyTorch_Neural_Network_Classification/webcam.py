import cv2
import numpy as np

# Face & Smile/Eye cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_roi = gray[y : y + h, x : x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ------------------- Eyes -------------------
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 10)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(
                frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2
            )

        # ------------------- Expression -------------------
        smiles = smile_cascade.detectMultiScale(face_roi, 1.7, 20)
        if len(smiles) > 0:
            expression = "Happy"
        elif len(eyes) == 0:
            expression = "Sleepy"
        else:
            expression = "Neutral"

        # ------------------- Age/Gender approximation -------------------
        # Quick trick: face width
        if w > 150:
            age = "Adult"
        else:
            age = "Child"

        # Quick trick: eyes + smile
        if len(eyes) > 1:
            gender = "Female"
        else:
            gender = "Male"

        # ------------------- Overlay -------------------
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
            f"Expression: {expression}",
            (x, y - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # Cyber glow and flow
        glow = frame.copy()
        cv2.rectangle(glow, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 255), 4)
        frame = cv2.addWeighted(glow, 0.3, frame, 0.7, 0)

    cv2.imshow("Fast Face AI", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
