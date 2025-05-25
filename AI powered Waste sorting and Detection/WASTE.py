
# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from serial_test import Send
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco/coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco/yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco/yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i- 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(0)
##vs = cv2.VideoCapture('C:\\Users\\My Lappy\\Desktop\\internship\\object detection\\')
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
                break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
                (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # probability is greater than the minimum probability
                        if confidence > args["confidence"]:
                                # scale the bounding box coordinates back relative to

                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top
                                # and and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates,
                                # confidences, and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                args["threshold"])

        # ensure at least one detection exists
        if len(idxs) > 0:
                count = 0
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        # draw a bounding box rectangle and label on the frame
                        text="{}".format(LABELS[classIDs[i]])
                        if text=="bottle" or text=="cup" or text == "bicycle" or text == "car" or text == "motorbike" or text == "aeroplane" or text == "bus" or text == "train" or text == "truck" or text == "boat" or text == "bench" or text == "backpack" or text == "umbrella" or text == "handbag" or text == "tie" or text == "suitcase" or text == "frisbee" or text == "skis" or text == "snowboard" or text == "sports ball" or text == "kite" or text == "skateboard" or text == "surfboard" or text == "tennis racket" or text == "bottle" or text == "wine glass" or text == "cup" or text == "fork" or text == "knife" or text == "spoon" or text == "bowl" or text == "chair" or text == "sofa" or text == "pottedplant" or text == "bed" or text == "diningtable" or text == "book" or text == "clock" or text == "vase" or text == "scissors" or text == "teddy bear" or text == "banana" or text == "apple" or text == "sandwich" or text == "orange" or text == "broccoli" or text == "carrot" or text == "hot dog" or text == "pizza" or text == "donut" or text == "cake" or text == "traffic light" or text == "fire hydrant" or text == "stop sign" or text == "parking meter" or text == "baseball bat" or text == "baseball glove":
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                text = "{}. {}: {:.4f}".format(count, LABELS[classIDs[i]],confidences[i])
                                cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                Send('O')

                #print('Total vehicle {}'.format(count))
        cv2.imshow('name',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# release the file pointers
print("[INFO] cleaning up...")
#writer.release()
vs.release()
