#importing libraries
import numpy as np
import cv2 
import random 
# Defing a list of classes that our model can predict
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
# defining random colors for each class that our model can predict
COLORS = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in CLASSES]
# reading in our model from file
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt","MobileNetSSD_deploy.caffemodel")
#reading the image on which we will perform object detection
img = cv2.imread("images/example_06.jpg")
#getting the height and width of the input image
(h,w) = img.shape[:2]
#ceating a blob out of the input image
blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),0.007843, (300, 300), 127.5)
# sending the blob to our neural net
net.setInput(blob)
# making the deetections
detections = net.forward()
# loooping over all the detections
for i in np.arange(0,detections.shape[2]):
	# grabbing the confidence  of our model in predictiing the object
	confidence = detections[0,0,i,2]
	# filtering out predictions with less confidence value
	if confidence > 0.5:
		id = detections[0,0,i,1] #getting the class id of a detected object
		box = detections[0,0,i,3:7] * np.array([w,h,w,h]) #defing our box
		(startX,startY,endX,endY) = box.astype("int") # naming each coordinate of our bounding box
		# drawing rctangles
		cv2.rectangle(img,(startX-1,startY-40),(endX+1,startY-3),COLORS[int(id)],-1) 
		cv2.rectangle(img,(startX,startY),(endX,endY),COLORS[int(id)],4)
		y = startY -15 if startY - 15 > 15 else startY + 15
		# adding label
		cv2.putText(img,CLASSES[int(id)],(startX + 10 ,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),1)

# displaying image
cv2.imshow("Output",img)
cv2.waitKey(0)
