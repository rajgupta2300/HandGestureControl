import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize camera
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5,
    model_complexity=1
)
drawing_utils = mp.solutions.drawing_utils

# To avoid multiple triggers in same frame
last_action_time = time.time()

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Optional: Digital zoom
    zoom_factor = 1.5
    cx, cy = width // 2, height // 2
    rx, ry = int(width / (2 * zoom_factor)), int(height / (2 * zoom_factor))
    cropped = frame[cy - ry:cy + ry, cx - rx:cx + rx]
    frame = cv2.resize(cropped, (width, height))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing_utils.draw_landmarks(frame, hand_landmarks)

            # Get thumb tip and index finger tip
            lm = hand_landmarks.landmark
            x1, y1 = int(lm[8].x * width), int(lm[8].y * height)  # index tip
            x2, y2 = int(lm[4].x * width), int(lm[4].y * height)  # thumb tip

            # Visuals
            cv2.circle(frame, (x1, y1), 10, (0, 255, 255), 3)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), 3)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Calculate distance
            distance = np.hypot(x2 - x1, y2 - y1)

            # Volume control logic
            now = time.time()
            if now - last_action_time > 0.2:  # cooldown 200ms
                if distance > 100:
                    pyautogui.press('volumeup', presses=3, interval=0.05)
                    last_action_time = now
                elif distance < 40:
                    pyautogui.press('volumedown', presses=3, interval=0.05)
                    last_action_time = now

            # Display distance
            cv2.putText(frame, f'Distance: {int(distance)}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("Hand Volume Control (pyautogui)", frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
