def crop_face(frame, bbox):
    """
    Safely crop face/person region from frame
    """
    x1, y1, x2, y2 = bbox
    h, w, _ = frame.shape

    # Clamp values (VERY IMPORTANT)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    cropped = frame[y1:y2, x1:x2]

    return cropped if cropped.size > 0 else None
