# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 20:17:59 2019

@author: omer
"""

import numpy as np
import cv2
import pickle

face_cascede = cv2.CascadeClassifier('cascades/data//haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
cap = cv2.VideoCapture(0)
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
    

while(True):
    
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascede.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    
    for(x,y,w,h) in faces:
     
        roi_gray=gray[y:y+h,x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h,x:x+w]
        
        # recognize        
        id_, conf = recognizer.predict(roi_gray)
        if conf >=45:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
    		
        img_item = "my-gray-image.png"
        cv2.imwrite(img_item,roi_gray)
        
        color = (255,0,0) #BGR
        stroke =2
        end_cord_x = x + w
        end_cord_y=y + h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
    
    cv2.imshow('frame',frame)
    
    if(cv2.waitKey(20) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()