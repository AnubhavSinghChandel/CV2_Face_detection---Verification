#   This is a just a small face recognition system 

    It just shows the UserID that is already present in the image of the user in front of the camera.
    Remeber the photo of user must be already be present in the Images folder.
    To run the system:
    1. Save image in the Images folder.
    2. Run the encodeGen.py file to generate the EncodeFile.p file.
    3. Finally run the face_rec file to fire up the web cam and use the system.

The system is for user verification for an e-voting system and uses libraries mentiomed below
1. OpenCV
2. face_recognition
3. cvzone
#   Installing Dependencies

    pip install opencv-python
    pip install face_recognition
    pickle already comes with python 3 distributions so no need to pip insatll that
