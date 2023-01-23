# right now each time the instance runs image_counter starts off as 0, but when connected to database we'll pull the userID and use it to store and verify image
import cv2
import face_recognition
import pickle

# Open the webcam
cap = cv2.VideoCapture(1)

img_counter = 0
# Capture an image from the webcam
while(1):
    ret, frame = cap.read()

    if not ret:
        print("Failed to caputre image")
        break
    cv2.imshow("UserImage", frame)

    key_press = cv2.waitKey(1)

    if key_press % 256 == 27:
        print("Quitting! Escape pressed")
        break
    elif key_press % 256 == 32:
        image_name = "Images/UserImage_{}.png".format(img_counter)
        cv2.imwrite(image_name, frame)
        print("UserImage registired")
        img_counter += 1
        cv2.destroyAllWindows()
        break

# Show the captured image
cv2.imshow("Captured Image", frame)
cv2.waitKey(0)


# Convert the image to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_recognition.face_locations(frame)
encodings = face_recognition.face_encodings(frame, faces)

# Get the encoding of the first face detected
encoding = encodings[0]

# Save the encoding to a file
with open("./Resources/User{}_encoding.txt".format(img_counter), "wb") as f:
    pickle.dump(encoding, f)

# Release the webcam and close the window

cv2.destroyAllWindows()
