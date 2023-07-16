import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getBiggestFace(faces):
    
    area = []
    i = 0
    for (x,y,w,h) in faces:
        area.append((w*h,i)) 
        i += 1

    area = sorted(area, reverse = True) #reverse = True sorts in descending order as we want the maximum area
    return area[0][1] 

user_name = input("Please enter your name.")
countFrames = 50
face_list = []

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) #flipping the frame about vertical axis
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #stores the gray scale image of the frame
    faces = classifier.detectMultiScale(gray)  #returns a lists of list of coordinates
    if(len(faces) > 0):
        face_index = getBiggestFace(faces) #index of face having maximum area bounding box
        face = faces[face_index]
        x, y, w, h = tuple(face)

        face = gray[y:y+h,x:x+w]  #coordinates of bounding box
        face = cv2.resize(face, (100,100)) #resizing to 100X100 pixels
        face = face.flatten()
        face_list.append(face)
        font = cv2.FONT_HERSHEY_SIMPLEX

        countFrames = countFrames - 1 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
        cv2.putText(frame, user_name, (x-15,y-15), font, 2, (255,0,0), 2, cv2.LINE_AA)
        
        
    
    cv2.imshow("Face Detection", frame)
    
    if(cv2.waitKey(1) > 30 or countFrames==0):
        break



face_list = np.array(face_list)
name_list = np.full((len(face_list), 1), user_name)
face_name_map = np.hstack([face_list, name_list]) #horizontally stacking two numpy arrays
data = []

if(os.path.exists("face_name_map.npy")):
    data = np.load("face_name_map.npy")
    data = np.vstack([data,face_name_map])
else:
    data = face_name_map

np.save("face_name_map.npy", data)
cap.release()
cv2.destroyAllWindows()
    