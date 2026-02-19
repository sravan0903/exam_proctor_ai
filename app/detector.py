from ultralytics import YOLO
import cv2

# Load YOLO model once (VERY IMPORTANT)
model = YOLO("yolov8n.pt")


def detect_faces(frame):
    """
    Detect faces/persons in a frame
    Returns number of detected persons
    """

    results = model(frame)

    face_count = 0
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # YOLO class 0 = person
            if cls == 0 and conf > 0.5:
                face_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(conf, 2)
                })

    return face_count, detections

def detect_objects(frame):
    results = model(frame)

    person_count = 0
    phone_count = 0
    person_boxes = []
    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # PERSON
            if cls == 0 and conf > 0.5:
                person_count += 1
                person_boxes.append((x1, y1, x2, y2))
                detections.append({
                    "type": "PERSON",
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

            # PHONE
            if cls == 67 and conf > 0.5:
                phone_count += 1
                detections.append({
                    "type": "PHONE",
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

    return person_count, phone_count, person_boxes, detections