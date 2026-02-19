import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_head_direction(frame):
    """
    Returns: CENTER / LEFT / RIGHT / UP / DOWN
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    # Key landmarks (nose tip, left cheek, right cheek)
    nose_tip = face_landmarks.landmark[1]
    left_cheek = face_landmarks.landmark[234]
    right_cheek = face_landmarks.landmark[454]

    nose_x = nose_tip.x * w
    left_x = left_cheek.x * w
    right_x = right_cheek.x * w

    center_x = (left_x + right_x) / 2
    offset = nose_x - center_x

    if offset > 15:
        return "RIGHT"
    elif offset < -15:
        return "LEFT"
    else:
        return "CENTER"
