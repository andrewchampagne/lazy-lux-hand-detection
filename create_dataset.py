import mediapipe as mp
import cv2 
import os
import matplotlib.pyplot as plt 
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

features = []
labels = []

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        features_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = hands.process(img_rgb)

        if res.multi_hand_landmarks:

            for hand_landmarks in res.multi_hand_landmarks:
                
                for i in range(len(hand_landmarks.landmark)):
                    print(hand_landmarks.landmark[i])

                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    features_aux.append(x)
                    features_aux.append(y)
            
            features.append(features_aux)
            labels.append(dir_)

        
f = open('data.pickle', 'wb')

pickle.dump({'features':features, 'labels':labels}, f)
f.close()