import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = tf.keras.models.load_model('sign_language_model.h5')

# Function to preprocess the cropped image for the model
def preprocess_image(cropped_image, target_size=(64, 64)):
    # Resize the image to match the model's input size 
    image_resized = cv2.resize(cropped_image, target_size)
    # Normalize the image pixels to [0, 1]
    image_normalized = image_resized / 255.0
    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

# Open webcam feed
cap = cv2.VideoCapture(0)

# Define padding 
padding_factor = 0.3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)
    
    # If hands are detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the bounding box for the hand (min and max points from the landmarks)
            min_x = min([landmark.x for landmark in landmarks.landmark])
            min_y = min([landmark.y for landmark in landmarks.landmark])
            max_x = max([landmark.x for landmark in landmarks.landmark])
            max_y = max([landmark.y for landmark in landmarks.landmark])

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            min_x_pixel = int(min_x * w)
            min_y_pixel = int(min_y * h)
            max_x_pixel = int(max_x * w)
            max_y_pixel = int(max_y * h)

            # Calculate the width and height of the bounding box
            box_width = max_x_pixel - min_x_pixel
            box_height = max_y_pixel - min_y_pixel

            # Add padding to the bounding box
            padding_x = int(box_width * padding_factor)
            padding_y = int(box_height * padding_factor)

            # Expand the bounding box with the padding, ensuring it stays within frame limits
            min_x_pixel = max(0, min_x_pixel - padding_x)
            min_y_pixel = max(0, min_y_pixel - padding_y)
            max_x_pixel = min(w, max_x_pixel + padding_x)
            max_y_pixel = min(h, max_y_pixel + padding_y)

            # Crop the hand region from the frame with padding
            cropped_hand = frame[min_y_pixel:max_y_pixel, min_x_pixel:max_x_pixel]

            # Preprocess the cropped hand image for prediction
            preprocessed_image = preprocess_image(cropped_hand)
            cv2.imshow("Preprocessed image", cropped_hand)

            # Feed the preprocessed image to the model for prediction
            predictions = model.predict(preprocessed_image)
            predicted_label = np.argmax(predictions, axis=1)

            # Display the predicted letter on the image
            cv2.putText(frame, f"Predicted Letter: {chr(predicted_label[0] + 65)}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the processed frame with hand landmarks and prediction
    cv2.imshow("Hand Detection and Prediction", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
