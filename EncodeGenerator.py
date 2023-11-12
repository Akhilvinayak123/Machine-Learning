import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import db


cred = credentials.Certificate("ServiceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://faceattendance-539ff-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendance-539ff.appspot.com"
})

#importing the employee images
folderPath = 'Images'
PathList = os.listdir(folderPath)
imgList = []

employeeIds = []
#print(PathList)
# for path is folderModePath:
#print(modePathList)
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    #print(path)
    #print(os.path.splitext(path)[0])
    employeeIds.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print(employeeIds)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList
print(" Encoding started... ")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, employeeIds]
print(encodeListKnown)
print("Encoding complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")



