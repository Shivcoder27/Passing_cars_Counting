import cvzone
from ultralytics import YOLO
import cv2
from sort import *
import numpy as np
#  # For WEBCAP
# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

  # FOR VEDIO
cap = cv2.VideoCapture("../vedios/cars.mp4")



model = YOLO('../Yolo-Weights/yolov8n.pt')  # Adjust path if needed

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]
mask = cv2.imread("mask.png")
#Tracking
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
# Run inference on the video feed
limits = [400,297,673,297]
totalCount = []
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    if not success:
        print("Failed to capture image.")
        break

    # Convert img to RGB format (YOLO expects RGB format)
    img_rgb = cv2.cvtColor(imgRegion, cv2.COLOR_BGR2RGB)

    # Save the image temporarily
    temp_image_path = 'temp_frame.jpg'
    cv2.imwrite(temp_image_path, img_rgb)

    # Use the image file path in predict
    results = model.predict(source=temp_image_path)  # Pass file path to model

    # The output of the model is a list of tensors. We process the first item.
    # Get the detections (boxes, confidences, and classes)
    pred = results[0].cpu().numpy()  # Convert the tensor to a numpy array

    detect = np.empty((0,5))

    # pred is a Nx6 array: [x1, y1, x2, y2, confidence, class_id]
    for detection in pred:
        x1, y1, x2, y2, conf, class_id = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to integers
        w, h = x2 - x1, y2 - y1



        # Get class name
        cls = int(class_id)  # Class ID is directly the last value in detection array
        currentClass = classNames[cls]
        if currentClass == "car" or currentClass == "truck" or currentClass == "bus"\
        or currentClass == "motorbike" and conf >0.3 :
            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
            #                    scale=0.6, thickness=1, offset=5)
            # Draw bounding box and confidence
            # cvzone.cornerRect(img, (x1, y1, w, h), l=10,rt = 5)
            conf = round(float(conf), 2)  # Confidence rounded to 2 decimals
            currentArray = np.array([x1,y1,x2,y2,conf])
            detect = np.vstack([detect, currentArray])  # Fixed stacking

    # Display the image
    # cv2.imshow("Image", img)
    resultsTracker = tracker.update(detect)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),scale=2,thickness=3,offset=5)
        cx,cy = x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),-1)
        if limits[0] <cx< limits[2] and limits[1]-40 <cy<limits[3]+40:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), 5)


    cvzone.putTextRect(img, f'{len(totalCount)}', (50,50))

    cv2.imshow("ImageRegion", img)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
