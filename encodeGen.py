import cv2
import os
import numpy as np
import face_recognition
import pickle

# Importing user images
folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []
userId = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    userId.append(os.path.splitext(path)[0])


def encodings(imageList):
    encodeList = []
    for img in imageList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding Images...")
encodeListKnown = encodings(imgList)
encodingListKnownWithIds = [encodeListKnown, userId]
print("Encoding Complete!")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodingListKnownWithIds, file)
file.close()
print("File Saved")
