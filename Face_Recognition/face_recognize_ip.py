
import cv2
import sys
import numpy
import os
import urllib.request

haar_file = 'haarcascade_frontalface_default.xml'
dataset = 'datasets'

url='http://192.168.0.101:8080/shot.jpg' #enter your ipcam url here

(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(dataset):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(dataset, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)

(images, labels) = [numpy.array(lis) for lis in [images, labels]]

model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
#webcam = cv2.Videocapture(0)
while True:
    #(_, im) = webcam.read()
    imgResp = urllib.request.urlopen(url)
    imgNp = numpy.array(bytearray(imgResp.read()),dtype=numpy.uint8)
    img = cv2.imdecode(imgNp,-1); frame = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        if prediction[1]<500:
            cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
        else:
            cv2.putText(frame,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))

    cv2.imshow('OpenCV', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
