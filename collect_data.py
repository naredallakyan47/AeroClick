import cv2
import mediapipe as mp
import pandas as pd
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = 'hand_landmarker.task'
output_path = '/home/user/PycharmProjects/Samsung/project/data/processed/gestures.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

data = []

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    time.sleep(1)

    start_time = time.time()
    print("✅ Ծրագիրը պատրաստ է (ՈՉ ՀԱՅԵԼԱՅԻՆ):")
    print("👉 Սեղմիր '0' բաց ձեռքի համար, '1' բռունցքի համար, 'q'՝ պահպանելու և դուրս գալու:")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue


        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        current_time_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, current_time_ms)

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                row = []
                for lm in hand_lms:
                    row.extend([lm.x, lm.y, lm.z])
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


                key = cv2.waitKey(1) & 0xFF
                if key == ord('0') or key == ord('1'):
                    label = chr(key)
                    data.append([label] + row)
                    print(f"📊 Գրանցվեց {label} | Քանակ՝ {len(data)}")

        cv2.imshow('Hand Collector (Normal View)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if data:
    df = pd.DataFrame(data)
    file_exists = os.path.isfile(output_path)
    df.to_csv(output_path, mode='a', header=not file_exists, index=False)
    print(f"\nՏվյալները պահպանվեցին: Ընդհանուր քանակ՝ {len(data)}")
else:
    print("\nՏվյալներ չեն հավաքվել:")