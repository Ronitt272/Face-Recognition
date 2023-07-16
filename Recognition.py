import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# training the model
k = 4
model = KNeighborsClassifier(4)
faces = np.load("face_name_map.npy")
# print(faces)
# print(faces.shape)

X = faces[:,0:len(faces[0])-1].astype(int)
y = faces[:,-1]

# print(X)
# print(y)

model.fit(X,y)
print(model)


# Model prediction

def getBiggestFace(faces):
    area = []
    i = 0
    for (x,y,w,h) in faces:
        area.append((w*h,i))
        i += 1

    area = sorted(area,reverse=True)
    return area[0][1]


classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray)

    if(len(faces) > 0):
        face_index = getBiggestFace(faces)
        # print(face_index)
        face = faces[face_index]
        x,y,w,h = face
        # slicing the image from gray scale
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100,100))
        face = face.flatten()
    
        # predicition
        result = model.predict([face])
        # print(result)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (x-10,y-10), (x+w, y+h), (0,0,255), 3) 
        cv2.putText(frame, result[0], (x-15,y-15), font, 2, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Face Recognition", frame)
    if(cv2.waitKey(1) > 30):
        break


cap.release()
cv2.destroyAllWindows()