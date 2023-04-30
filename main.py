# right now each time the instance runs image_counter starts off as 0, but when connected to database we'll pull the userID and use it to store and verify image
import os
import cv2
import pickle
import face_recognition
import numpy as np
import cvzone

# Open the webcam
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/Background.png')

# # Importing mode images
# modePath = 'Resources/Modes'
# modePathList = os.listdir(modePath)
# imgModeList = []
# for path in modePathList:
#     imgModeList.append(cv2.imread(os.path.join(modePath, path)))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
print("Encode File loaded successfully!")
file.close()
encodeListKnown, userIds = encodeListKnownWithIds

# Capture an image from the webcam
while(1):
    ret, frame = cap.read()

    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162+480, 55:55+640] = frame
    # imgBackground[44:44+633, 808:808+414] = imgModeList[0]

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(
                encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(
                encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                cv2.putText(imgBackground, f"{userIds[matchIndex]}", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

    cv2.imshow("UserImage", imgBackground)
    if(cv2.waitKey(1) == 27):
        break
