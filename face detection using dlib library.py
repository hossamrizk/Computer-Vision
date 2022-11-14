import dlib
import cv2
import os
os.getcwd()
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap=cv2.VideoCapture(0)
while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    faces=detector(gray)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        landmarks=predictor(gray,face)
        for n in range(0,68):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            print(x,y)
            cv2.circle(frame,(x,y),3,(255,0,0),-1)
    cv2.imshow("Face detection",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release() 
cv2.destroyAllWindows()       