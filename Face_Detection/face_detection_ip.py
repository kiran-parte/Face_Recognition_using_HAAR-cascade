
import cv2
import urllib.request
import numpy as np
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

url='http://192.168.0.101:8080/shot.jpg' #add your ipcam url here

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[x:x+w, y:y+h]
        roi_frame = frame[x:x+w, y:y+h]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame


while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img = cv2.imdecode(imgNp,-1) 
	frame = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('video', canvas)
	#time.sleep(0.1)
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
		
video_cap.release()
cv2.destroyAllWindows()

