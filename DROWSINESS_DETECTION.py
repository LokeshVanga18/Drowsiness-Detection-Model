import cv2
import imutils as i
import dlib
import numpy as np
from scipy.spatial import distance as d
from imutils import face_utils
import winsound 
import time

c = 0
blink = 0

sh = "C_fold/shape_predictor_68_face_landmarks.dat"

det = dlib.get_frontal_face_detector()
pre = dlib.shape_predictor(sh)

(lx , ly) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rx , ry) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

cam = cv2.VideoCapture(0)

def ear(eye):
    A = d.euclidean(eye[1] , eye[5])
    B = d.euclidean(eye[2] , eye[4])
    C = d.euclidean(eye[0] , eye[3])

    return (A+B) / (2.0 * C)

while True:
    _ , img = cam.read()
    img = i.resize(img , width=700)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    rects = det(gray , 0)
    print(rects)

    for rect in rects:
        shape = pre(gray , rect)
        shape = face_utils.shape_to_np(shape)

        leye = shape[lx:ly]
        reye = shape[rx:ry]

        lear = ear(leye)
        rear = ear(reye)

        e = (lear+rear) / 2.0

        lhull = cv2.convexHull(leye)
        rhull = cv2.convexHull(reye)

        cv2.drawContours(img , [lhull] , 0 , (0,255,0) , 1)
        cv2.drawContours(img , [rhull] , 0 , (0,255,255) , 1)

        if e<0.2:
            c += 1
            if c>30:
                cv2.putText(img , "WAKE UP BOY" , (10,30) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,255,0) , 2)
                winsound.Beep(500,1000)
            elif c>0 and c<2:
                blink +=1

        else:
            c=0
    cv2.putText(img , f"BLINKS -->{blink}" , (10,100) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,0,255) , 2)
    cv2.imshow("lokesh" , img)    
    key = cv2.waitKey(1)
    if key == ord('l'):
        break

cam.release()
cv2.destroyAllWindows()