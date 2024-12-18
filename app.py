from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import pickle
import pyttsx3  # Import pyttsx3 for text-to-speech
import mediapipe as mp


app = Flask(__name__)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7',
               34: '8', 35: '9', 36: 'SPACE', 37: 'BACKSPACE', 38: 'READ'}

current_detected = ""
current_alphabet = None
start_time = None
time_threshold = 1.5  # Adjust the threshold as needed

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)
            frame = process_frame(frame)
            ret, buffer  = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def process_frame(frame):
    global current_alphabet, current_detected, start_time  # Declare global variables

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * w) - 10
        y1 = int(min(y_) * h) - 10

        x2 = int(max(x_) * w) - 10
        y2 = int(max(y_) * h) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        if predicted_character != current_alphabet:
            current_alphabet = predicted_character
            start_time = time.time()

        if time.time() - start_time > time_threshold:
            if current_alphabet == 'READ':
                speak_text(current_detected)
                current_detected += ' '  # Add space to separate detected words
            elif current_alphabet == 'SPACE':
                speak_text("SPACE")
                current_detected += ' '
            elif current_alphabet == 'BACKSPACE':
                current_detected = current_detected[:-1]
                speak_text("BACKSPACE")
            else:
                speak_text(current_alphabet)
                current_detected += current_alphabet

    return frame

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/subtitles')
def subtitles():
    return current_detected

if __name__ == '__main__':
    app.run(debug=True)
