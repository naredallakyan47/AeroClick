import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'hand_landmarker.task'
ai_model = tf.keras.models.load_model('hand_model.h5')

screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

frame_reduction = 100
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        cv2.rectangle(frame, (frame_reduction, frame_reduction),
                      (w - frame_reduction, h - frame_reduction), (255, 0, 255), 2)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        current_time_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, current_time_ms)

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                coords = []
                for lm in hand_lms:
                    coords.extend([lm.x, lm.y, lm.z])

                prediction = ai_model.predict(np.array([coords]), verbose=0)
                gesture = np.argmax(prediction)

                idx_tip = hand_lms[8]
                x1, y1 = idx_tip.x * w, idx_tip.y * h

                x3 = np.interp(x1, (frame_reduction, w - frame_reduction), (0, screen_width))
                y3 = np.interp(y1, (frame_reduction, h - frame_reduction), (0, screen_height))

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                if gesture == 0:
                    pyautogui.moveTo(clocX, clocY, _pause=False)
                    plocX, plocY = clocX, clocY
                    cv2.putText(frame, "MOVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif gesture == 1:
                    pyautogui.click(button='right')
                    cv2.putText(frame, "RIGHT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    time.sleep(0.2)

        cv2.imshow('AeroPoint Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()