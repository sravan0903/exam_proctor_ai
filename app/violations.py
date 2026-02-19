import time

LOOK_AWAY_LIMIT = 3  # seconds

last_looking_center = {}


def check_looking_away(student_email, direction):
    """
    Returns True if looking away for too long
    """
    now = time.time()

    if direction == "CENTER":
        last_looking_center[student_email] = now
        return False

    last_time = last_looking_center.get(student_email, now)
    duration = now - last_time

    if duration >= LOOK_AWAY_LIMIT:
        return True

    return False
