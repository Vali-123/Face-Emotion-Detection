import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('C:/Users/Topiv/Downloads/NLP CODE/projectdeeplearning/model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the face detector (Haar Cascade)
face_classifier = cv2.CascadeClassifier(r'C:/Users/Topiv/Downloads/NLP CODE/projectdeeplearning/haarcascade_frontalface_default (1).xml')


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Region of interest (ROI) for the face in grayscale
        roi_gray = gray[y:y + h, x:x + w]

        # Resize the face to 48x48 (model input size)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Preprocess the face image for prediction
        if np.sum(roi_gray) != 0:  # If there's a face
            roi = roi_gray.astype('float32') / 255.0  # Normalize the pixel values
            roi = img_to_array(roi)  # Convert to an array
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            # Make a prediction using the model
            prediction = model.predict(roi)[0]

            # Get the emotion with the highest probability
            label = emotion_labels[prediction.argmax()]

            # Position to place the label
            label_position = (x, y - 10)

            # Put the label text on the frame
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            # If no face is detected, display a message
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
