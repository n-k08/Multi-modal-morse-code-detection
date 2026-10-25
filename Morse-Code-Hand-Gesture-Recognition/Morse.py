import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Morse code sequence and translated text
morse_code_sequence = ''
translated_text = ''
gesture_state = "neutral"

# Define pause duration, debounce time, and threshold
pause_duration = 4  # Pause duration in seconds
debounce_time = 2  # Debounce time in seconds
threshold = 0.4  # Threshold for detecting a fist (dash)

# Initialize last gesture time and last gesture end time
last_gesture_time = time.time()  
last_gesture_end_time = time.time()  

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing Utility
mp_drawing = mp.solutions.drawing_utils

# Morse code dictionary
morse_code_dict = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F', '--.': 'G', '....': 'H',
    '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P',
    '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z',
    '.----': '1', '..---': '2', '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7',
    '---..': '8', '----.': '9', '-----': '0',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-.-.--': '!', '-..-.': '/', '-.--.': '(',
    '-.--.-': ')', '.-...': '&', '---...': ':', '-.-.-.': ';', '-...-': '=', '.-.-.': '+', '-....-': '-',
    '..--.-': '_', '.-..-.': '"', '...-..-': '$', '.--.-.': '@', '/': ' '
}

def text_to_morse(text):
    """
    Converts text to Morse code.

    Args:
        text (str): The text to convert.

    Returns:
        str: The converted Morse code.
    """
    morse = ''
    reverse_morse_code = {
        character: sequence
        for sequence, character in morse_code_dict.items()
        if len(character) == 1
    }
    for letter in text:
        if letter.upper() in reverse_morse_code:
            morse += reverse_morse_code[letter.upper()] + ' '
        elif letter == ' ':
            morse += '/ '
        else:
            morse += '? '  # Handle unrecognized characters
    return morse

def start_webcam():
    """
    Starts the webcam and performs hand gesture recognition.

    This function reads frames from the webcam, processes them using MediaPipe Hands,
    and performs hand gesture recognition based on the distances between hand landmarks.
    It displays the recognized gesture, translated text, and input sequence on the frame.

    Returns:
        None
    """
    global last_gesture_time, last_gesture_end_time, translated_text, morse_code_sequence
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip and convert the frame to RGB
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Hands
            results = hands.process(frame)

            # Convert the frame back to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            max_distance = 0  # Initialize max_distance here

            # Draw hand landmarks and calculate distances
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Calculate distances between hand landmarks
                    wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                                      hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])
                    thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                          hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
                    index_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                          hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
                    middle_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                           hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
                    ring_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                                         hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y])
                    pinky_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                                          hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y])

                    thumb_distance = np.linalg.norm(thumb_tip - wrist)
                    index_distance = np.linalg.norm(index_tip - wrist)
                    middle_distance = np.linalg.norm(middle_tip - wrist)
                    ring_distance = np.linalg.norm(ring_tip - wrist)
                    pinky_distance = np.linalg.norm(pinky_tip - wrist)

                    current_max_distance = max([thumb_distance, index_distance, middle_distance, ring_distance, pinky_distance])
                    if current_max_distance > max_distance:
                        max_distance = current_max_distance

            current_time = time.time()  # Get the current time.

            if current_time - last_gesture_time > debounce_time:  # Check if the debounce period has passed.
                if results.multi_hand_landmarks:
                    if max_distance > threshold:
                        new_gesture = "dash"
                    else:
                        new_gesture = "dot"
                else:
                    new_gesture = "neutral"
                
                cv2.putText(frame, f"Gesture: {new_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if new_gesture != "neutral":
                    if new_gesture == "dot":
                        morse_code_sequence += '.'
                    elif new_gesture == "dash":
                        morse_code_sequence += '-'
                    last_gesture_time = current_time  # Update last_gesture_time when a gesture ends
                
                if current_time - last_gesture_time > pause_duration and morse_code_sequence:  # Check if a pause is detected
                    translated_text += morse_code_dict.get(morse_code_sequence, '?')
                    morse_code_sequence = ''
                    last_gesture_time = current_time  # Reset last_gesture_time after translating Morse code sequence

            morse_code_text = text_to_morse(translated_text)

            cv2.putText(frame, f"Translated Text: {translated_text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Input Sequence: {morse_code_sequence}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    start_webcam()