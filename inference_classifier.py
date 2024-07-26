import mediapipe as mp
import cv2 
import numpy as np
import pickle
import keyboard
import time
import signal
import os

STATUS_FILE = "status.txt"
FLAG_FILE = "stop_flag.txt"
# Global termination flag


# Function to handle termination signals
def signal_handler(signum, frame):
    global should_stop
    should_stop = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


last_press_time = time.time()
press_interval = 1.0 / 4  # Interval between key presses (4 times per second)


def press_key(key):
    global last_press_time
    current_time = time.time()
    
    if current_time - last_press_time >= press_interval:
        print("key Pressed")
        keyboard.press(key)
        time.sleep(0.05)
        keyboard.release(key)
        last_press_time = current_time

dict = pickle.load(open('./model.p', 'rb'))
model = dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

ability_keys_with_default = {0: 'q', 1: 'w', 2: 'd', 3: 'r', 4: 'e', 5: 'f', 6 : 'None'}
ability_keys = {0: 'q', 1: 'w', 2: 'd', 3: 'r', 4: 'e', 5: 'f'}

with open(STATUS_FILE, "w") as f:
    f.write("ready")

cap = cv2.VideoCapture(0)

while True:
    if os.path.exists(FLAG_FILE):
        os.remove(FLAG_FILE)
        break
    features_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res = hands.process(frame_rgb)

    if res.multi_hand_landmarks:

        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in res.multi_hand_landmarks:
                
                for i in range(len(hand_landmarks.landmark)):
                    

                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    features_aux.append(x)
                    features_aux.append(y)
                    x_.append(x)
                    y_.append(y)

        if len(features_aux) != 42:
             continue 
        
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        prediction = model.predict([np.asarray(features_aux)])
        predicted_key = ability_keys_with_default[int(prediction[0])]

        
        if predicted_key in ability_keys.values():
             press_key(str(predicted_key))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(frame, f"Pressed Key: {predicted_key}", (x1 - 50, y1), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    

if os.path.exists(STATUS_FILE):
    os.remove(STATUS_FILE)
cap.release()

