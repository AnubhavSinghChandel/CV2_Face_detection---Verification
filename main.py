import cv2
import face_recognition
import pickle

# Load the saved encoding from the file
with open("./Resources/User1_encoding.txt", "rb") as f:
    known_encoding = pickle.load(f)

# Open the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture a frame from the webcam
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, faces)

    # For each detected face
    for encoding in encodings:
        # Compare the encoding with the known encoding
        dist = face_recognition.face_distance([known_encoding], encoding)
        if dist < 0.6:
            cv2.putText(frame, 'Verified', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Not Verified', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Webcam', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the
cap.release()
cv2.destroyAllWindows()
