import cv2
import numpy as np
cap = cv2.VideoCapture(0)
##### Parametes which are used our Project

whT = 320
confThreshold = 0.5
nms_threshold = 0.3
classesfile = "coco.names.txt" # coco data set
classnames = [] # create a empty list for names

with open(classesfile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n') # extract the names form coco file in our classnames list


modelConfiguration = 'yolov3.cfg.txt' # load our yolo configure file
modelWeights = 'yolov3.weights' # load our yolo weights and simply say CNN models
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights) # to start our network
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # set preferable backend as opencv
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # set our preferable target(processor) as CPU

def findObjects(outputs, img):
    hT, wT, cT = img.shape # here we collect width, height from the image
    bbox = [] # create a bbox empty list
    classIds = [] # crete classIds empty list
    confs = [] # create a list for saving the confidence value of image
    for output in outputs: # this loop collect the imformation from outputs and outpusts is nothing but these are the
        # boxe plotted on our imgages so that it detect the right object for if 300 boxes plotted on our image then
        # every boxes detect object thats why when the object is moved then the accuray is also changed with movement.
        for det in output: # det is like a row of 85 columns

            scores = det[5:] # here we removed our starting 5 columns bcz it contain only x,y,width,height and confidence

            classId = np.argmax(scores) # this give index value of highest number
            confidence = scores[classId] # in confidence variable we save that highest number
            if confidence > confThreshold: # give a minimum probality value
                w, h = int(det[2]*wT), int(det[3]*hT) # detect height and width from det variable
                x, y =int((det[0]*wT)-w/2), int((det[1]*hT)-h/2) # detect x and y from det variable
                bbox.append([x,y,w,h]) # append geometrical imformation in bbox list bcz it change with movement of object
                classIds.append(classId)
                confs.append(float(confidence))
    indeces = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nms_threshold) # here we use NMABoxes so that more than one rectangular
    # box not plotted on our image at same time

    for i in indeces: # here are the index value of our classes like for person the index value is 1 and so on
        # and it is measure by bbox bcz if put our face imgae to our web cam then it recognise it as a person
        # and set the indeces value and there are many object in one frmae so that we looping in our indeces and
        # it recognise every object and tell us.

        i = i[0] # the index value is saved in 2D array format so we access first element.

        box = bbox[i] # here we put the geometrical value according index number bcz bbox is a list which contain
        # geometrical value

        x,y,w,h = box[0],box[1],box[2],box[3] # here we set the cordinate for plot rectangle on our object.
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img,f'{classnames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),
                           cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False) # to give our image in blob format to our
    # network by the use of blobFormImage function
    net.setInput(blob) # set our Input in blob format
    layers = net.getLayerNames() # our network use multiple layers simply called as hidden layers

    outputName = [layers[i[0]-1] for i in net.getUnconnectedOutLayers()] # output index start from 1 so we substract 1
    outputs = net.forward(outputName) # here outputName are the names of output Hidden layes while outputs are the list
    # of each class imformation

    findObjects(outputs,img) # this the function which find the object
    cv2.imshow("Image",img)

    cv2.waitKey(1)