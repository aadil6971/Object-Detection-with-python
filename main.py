import numpy as np
import cv2 
import random 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in CLASSES]

net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")

img = cv2.imread("images/example_11.jpg")
(h,w) = img.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),0.007843, (300, 300), 127.5)

net.setInput(blob)

detections = net.forward()

for i in np.arange(0,detections.shape[2]):
	confidence = detections[0,0,i,2]
	if confidence > 0.5:
		id = detections[0,0,i,1]
		box = detections[0,0,i,3:7] * np.array([w,h,w,h])
		(startX,startY,endX,endY) = box.astype("int")

		cv2.rectangle(img,(startX-1,startY-40),(endX+1,startY-3),COLORS[int(id)],-1)
		cv2.rectangle(img,(startX,startY),(endX,endY),COLORS[int(id)],4)
		y = startY -15 if startY - 15 > 15 else startY + 15
		cv2.putText(img,CLASSES[int(id)],(startX + 10 ,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),1)


cv2.imshow("Output",img)
cv2.waitKey(0)
