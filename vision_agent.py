import cv2
from eye_detector import get_eye_landmarks
from ear_calculator import calculate_ear

class VisionAgent:

    def __init__(self):
        self.EAR_THRESHOLD = 0.25
        self.counter = 0
        self.frame_limit = 20

    def process_frame(self, frame, landmarks, frame_w, frame_h):

        left_eye = get_eye_landmarks(landmarks, [33,160,158,133,153,144], frame_w, frame_h)
        right_eye = get_eye_landmarks(landmarks, [362,385,387,263,373,380], frame_w, frame_h)

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        ear = (left_ear + right_ear) / 2

        if ear < self.EAR_THRESHOLD:
            self.counter += 1
        else:
            self.counter = 0

        return self.counter