import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)
# Improved yawning detection logic
cap = cv2.VideoCapture(0)

sleep_counter = 0
yawn_counter = 0

# Store MAR history for smoothing
mar_history = deque(maxlen=5)

# Adaptive baseline
baseline_mar = None


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye):
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def calculate_mar(mouth):
    A = distance(mouth[2], mouth[6])
    B = distance(mouth[3], mouth[5])
    C = distance(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            # LEFT EYE
            left_eye_ids = [33, 160, 158, 133, 153, 144]
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in left_eye_ids]

            # RIGHT EYE
            right_eye_ids = [362, 385, 387, 263, 373, 380]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in right_eye_ids]

            # MOUTH
            mouth_ids = [78, 81, 13, 311, 308, 402, 14]
            mouth = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_ids]

            # DRAW POINTS
            for p in left_eye + right_eye:
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

            for p in mouth:
                cv2.circle(frame, p, 2, (255, 255, 0), -1)

            # CALCULATIONS
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2
            mar = calculate_mar(mouth)

            # ----------- FRAME AVERAGING (SMOOTHING) -----------
            mar_history.append(mar)
            smoothed_mar = np.mean(mar_history)

            # ----------- ADAPTIVE THRESHOLD -----------
            if baseline_mar is None:
                baseline_mar = smoothed_mar
            else:
                baseline_mar = 0.9 * baseline_mar + 0.1 * smoothed_mar

            mar_threshold = baseline_mar + 0.15

            # ----------- DROWSINESS (EAR) -----------
            if ear < 0.25:
                sleep_counter += 1
            else:
                sleep_counter = 0

            if sleep_counter > 20:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # ----------- TEMPORAL ANALYSIS FOR YAWN -----------
            if smoothed_mar > mar_threshold:
                yawn_counter += 1
            else:
                # Detect only if sustained opening (not talking)
                if yawn_counter > 15:
                    cv2.putText(frame, "YAWN DETECTED", (50, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                yawn_counter = 0

            # DEBUG INFO
            cv2.putText(frame, f"MAR: {smoothed_mar:.2f}", (400, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Thresh: {mar_threshold:.2f}", (400, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()