import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

sleep_counter = 0


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye):

    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


def calculate_mar(mouth):

    A = distance(mouth[2], mouth[6])
    B = distance(mouth[3], mouth[5])
    C = distance(mouth[0], mouth[4])

    mar = (A + B) / (2.0 * C)
    return mar


while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            print("Face detected")

            landmarks = face_landmarks.landmark

            # LEFT EYE
            left_eye_ids = [33, 160, 158, 133, 153, 144]

            left_eye = []
            for i in left_eye_ids:
                x = int(landmarks[i].x * w)
                y = int(landmarks[i].y * h)
                left_eye.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # RIGHT EYE
            right_eye_ids = [362, 385, 387, 263, 373, 380]

            right_eye = []
            for i in right_eye_ids:
                x = int(landmarks[i].x * w)
                y = int(landmarks[i].y * h)
                right_eye.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # MOUTH
            mouth_ids = [78, 81, 13, 311, 308, 402, 14]

            mouth = []
            for i in mouth_ids:
                x = int(landmarks[i].x * w)
                y = int(landmarks[i].y * h)
                mouth.append((x, y))
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

            # CALCULATIONS
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)

            ear = (ear_left + ear_right) / 2

            mar = calculate_mar(mouth)

            # DROWSINESS CHECK
            if ear < 0.25:
                sleep_counter += 1
            else:
                sleep_counter = 0

            if sleep_counter > 20:
                cv2.putText(frame,
                            "DROWSINESS ALERT!",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

            # YAWNING
            if mar > 0.6:
                cv2.putText(frame,
                            "Yawning Detected",
                            (50, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2)

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()