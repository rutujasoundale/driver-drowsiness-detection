def get_eye_landmarks(landmarks, eye_indices, width, height):
    points = []

    for i in eye_indices:
        x = int(landmarks[i].x * width)
        y = int(landmarks[i].y * height)
        points.append((x, y))

    return points