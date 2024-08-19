import cv2
import numpy as np

# initialize camera
cap = cv2.VideoCapture(0)
# define camera parameters
FOV = 62.2 # camera field of view
F = 3.04 # focal length of camera in mm
IMAGE_WIDTH = 640 # width of camera image
IMAGE_HEIGHT = 480 # height of camera image

# calculate camera constant
CAMERA_CONSTANT = (FOV/2)/np.tan((F/2)/1000)

classNames = []
classFile = "coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo

while True:
    success, img = cap.read()
    result, objectInfo = getObjects(img,0.45,0.2, objects=["cell phone", "backpack", 'dining table', 'toothbrush','bottle'])
    for obj in objectInfo:
        box, className = obj
        # calculate the distance of the object from the camera
        x, y, w, h = box
        distance = (w + h) * CAMERA_CONSTANT / 2
        print(className, "distance:", distance)
        
    cv2.imshow("Output",img)
    cv2.waitKey(1)

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()