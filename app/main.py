from app.detector import detect_faces, detect_objects
from fastapi import FastAPI, Form, UploadFile, File
import cv2
import numpy as np
import os
from app.recognition import extract_face_embedding
from app.recognition import is_same_person
from app.utils import crop_face
from app.head_pose import get_head_direction
from app.violations import check_looking_away


REGISTER_DIR = "models/registered_faces"
os.makedirs(REGISTER_DIR, exist_ok=True)

app = FastAPI(
    title="AI Exam Proctoring Service",
    description="Detects exam violations using AI",
    version="1.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "AI Proctoring Service is running",
        "message": "Ready to analyze exam frames"
    }

@app.get("/health")
def health_check():
    return {"health": "OK"}


@app.post("/analyze-frame")
async def analyze_frame(
    studentEmail: str= Form(...),
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    person_count, phone_count, person_boxes, detections = detect_objects(frame)

    violation = None
    similarity = None
    headDirection = None

    # 🔴 HIGHEST PRIORITY: PHONE
    if phone_count > 0:
        violation = "PHONE_DETECTED"

    # 🔴 NO PERSON
    elif person_count == 0:
        violation = "NO_FACE_DETECTED"

    # 🔴 MULTIPLE PEOPLE
    elif person_count > 1:
        violation = "MULTIPLE_FACES"

    # 🟢 SINGLE PERSON → FACE + HEAD CHECK
    else:
        face_crop = crop_face(frame, person_boxes[0])

        if face_crop is None:
            violation = "FACE_NOT_CLEAR"
        else:
            # 🔹 HEAD MOVEMENT CHECK (MediaPipe)
            headDirection = get_head_direction(face_crop)

            if headDirection and headDirection != "CENTER":
                looking_away = check_looking_away(
                    studentEmail, headDirection
                )

                if looking_away:
                    violation = "LOOKING_AWAY"

            # 🔹 FACE RECOGNITION (only if no look-away violation yet)
            if violation is None:
                stored_path = f"{REGISTER_DIR}/{studentEmail}.npy"

                if os.path.exists(stored_path):
                    stored_embedding = np.load(stored_path)
                    live_embedding = extract_face_embedding(face_crop)

                    match, similarity = is_same_person(
                        live_embedding, stored_embedding
                    )

                    if not match:
                        violation = "IMPERSONATION"
                else:
                    violation = "FACE_NOT_REGISTERED"

    return {
        "personsDetected": int(person_count),
        "phonesDetected": int(phone_count),
        "headDirection": headDirection,
        "violation": violation,
        "similarity": float(similarity) if similarity is not None else None,
        "detections": detections
    }



@app.post("/register-face")
async def register_face(studentEmail: str= Form(...), file: UploadFile = File(...)):

    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    embedding = extract_face_embedding(frame)

    file_path = f"{REGISTER_DIR}/{studentEmail}.npy"
    np.save(file_path, embedding)

    return {
        "status": "Face registered successfully",
        "studentEmail": studentEmail
    }