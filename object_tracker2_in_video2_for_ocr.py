import cv2
import numpy as np
import time
import sys
import os
import argparse
import imutils
import base64
from scipy.spatial import distance
import datetime
import mysql.connector
import easyocr
import os
import cv2

CONFIDENCE = 0.3
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


mydb = mysql.connector.connect(
  host="slotracha168.com",
  user="slotrac_root",
  password="npsd1234",
  database="nextsoft_1"
)

mycursor = mydb.cursor()
sql = "DELETE FROM image_lpr WHERE num_ch > '%d'" % (1)
mycursor.execute(sql)
mydb.commit()
print(mycursor.rowcount, "record(s) deleted image_lpr")

# the neural network configuration
config_path = "config_lpr/darknet-yolov3.cfg"
# the YOLO net weights file
weights_path = "config_lpr/model.weights"
# weights_path = "weights/yolov3-tiny.weights"

# loading all the class labels (objects)
labels = open("config_lpr/classes.names").read().strip().split("\n")
# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

cap = cv2.VideoCapture('rw_fail.mp4') #2021-09-18 18-04-52 , 2021-09-17 18-28-00
counts=0
ccc=0
while(cap.isOpened()):

    current_time = datetime.datetime.now()
    secc = current_time.second
    minu = current_time.minute
    hh = current_time.hour
    dd = current_time.day
    mm = current_time.month
    yy = current_time.year
    text_day = 'Day:'+str(dd)+'-'+str(mm)+'-'+str(yy)
    text_time = 'Time:'+str(hh)+':'+str(minu)+':'+str(secc)


    ccc=ccc+1
    ret, frame1 = cap.read()

    width1 = 250
    height1 = 250 # keep original height
    dim1 = (width1, height1)
        
    # resize image
    frame1_input = cv2.resize(frame1, dim1, interpolation = cv2.INTER_AREA)

    width = 960
    height = 540 # keep original height
    dim = (width, height)
    
    # resize image
    frame1 = cv2.resize(frame1, dim, interpolation = cv2.INTER_AREA)
    frame20=frame1
    ########################################################
    ########################## 1. color detection #####################
    ########################################################
    ########################################################
    frame = frame1[1:500,400:600]
    frame2=frame
    frame3=frame
    frame4=frame
    hsv_frame = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)
    # Red color
    low_red = np.array([160, 50, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    kernel = np.ones((5,5),np.uint8)
    rw = cv2.erode(red_mask,kernel,iterations = 1)
    #cv2.imshow('red_mask', rw)


    ########################################################
    ########################## 2. Determine for teb road ##
    ########################################################
    ########################################################
    print('sum(sum(rw))', sum(sum(rw)))
    if sum(sum(rw)) >= 5000:
        lim_dist = 6
        ch = 0
        thres=120
        size_lp = 26
        text_tab = 'ขาว-แดง'
        road = 'ถนนสรรพาวุธ'
        cv2.putText(frame1, 'Sanphawut Road', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame1,text_day, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame1,text_time, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame1,'red-white', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        tab_num = 0

        # Get width and height of video
        w = cap.get(3)
        h = cap.get(4)
        #print('w,h',w,h)
        frameArea = h * w
        areaTH = frameArea / 400
        # Lines
        line_up = int(2 * (h / 5))
        line_down = int(3 * (h / 5))

        up_limit = int(1 * (h / 5))
        down_limit = int(4 * (h / 5))
        #print("Red line y:", str(line_down))
        #print("Blue line y:", str(line_up))
        line_down_color = (255, 0, 0)
        line_up_color = (255, 0, 255)
        pt1 = [0, line_down]
        pt2 = [w, line_down]
        pts_L1 = np.array([pt1, pt2], np.int32)
        pts_L1 = pts_L1.reshape((-1, 1, 2))
        pt3 = [0, line_up]
        pt4 = [w, line_up]
        pts_L2 = np.array([pt3, pt4], np.int32)
        pts_L2 = pts_L2.reshape((-1, 1, 2))

        pt5 = [0, up_limit]
        pt6 = [w, up_limit]
        pts_L3 = np.array([pt5, pt6], np.int32)
        pts_L3 = pts_L3.reshape((-1, 1, 2))
        pt7 = [0, down_limit]
        pt8 = [w, down_limit]
        pts_L4 = np.array([pt7, pt8], np.int32)
        pts_L4 = pts_L4.reshape((-1, 1, 2))

        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (650, 650)
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (220, 350)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        cv2.rectangle(frame1, start_point, end_point, color, thickness)


    else:
        lim_dist = 4
        ch = 1
        thres=100
        size_lp = 21
        frame5 = frame1
        text_tab = 'ขาว-เหลือง'
        road = 'ถนนกรุงเทพ'
        tab_num = 1
        cv2.putText(frame1, 'BKK Road', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame1,text_day, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame1,text_time, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame1,'yellow-white', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        # Get width and height of video
        w = cap.get(3)
        h = cap.get(4)-80
        #print('w,h',w,h)
        frameArea = h * w
        areaTH = frameArea / 400
        # Lines
        line_up = int(2 * (h / 5))
        line_down = int(3 * (h / 5))
        up_limit = int(1 * (h / 5))
        down_limit = int(4 * (h / 5))
        #print("Red line y:", str(line_down))
        #print("Blue line y:", str(line_up))
        line_down_color = (255, 0, 0)
        line_up_color = (255, 0, 255)
        pt1 = [0, line_down]
        pt2 = [w, line_down]
        pts_L1 = np.array([pt1, pt2], np.int32)
        pts_L1 = pts_L1.reshape((-1, 1, 2))
        pt3 = [0, line_up]
        pt4 = [w, line_up]
        pts_L2 = np.array([pt3, pt4], np.int32)
        pts_L2 = pts_L2.reshape((-1, 1, 2))
        pt5 = [0, up_limit]
        pt6 = [w, up_limit]
        pts_L3 = np.array([pt5, pt6], np.int32)
        pts_L3 = pts_L3.reshape((-1, 1, 2))
        pt7 = [0, down_limit]
        pt8 = [w, down_limit]
        pts_L4 = np.array([pt7, pt8], np.int32)
        pts_L4 = pts_L4.reshape((-1, 1, 2))

        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (650, 650)
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (450, 350)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        cv2.rectangle(frame1, start_point, end_point, color, thickness)
    
    cv2.imshow('frame1_input',frame1_input)
    cv2.imwrite('car_input22.jpg',frame1_input)
    # Open a file in binary mode
    file2 = open('car_input22.jpg','rb').read()
    # We must encode the file to get base64 string
    file2 = base64.b64encode(file2)
    
    mycursor = mydb.cursor()
    sql = "INSERT INTO image_lpr (photo, num_ch) VALUES (%s,%s)"
    val = (file2,ccc)
    mycursor.execute(sql, val)
    mydb.commit()
    print(mycursor.rowcount, "record file2 inserted.")
    
    '''
    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
      
    # resize image
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    '''
    #####################################################################
    ################ 3. Find location of license plate ##################
    ######################################################################
    ret, frame = cap.read()
    image = frame
    h, w = image.shape[:2]
    # create 4D blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    #print("image.shape:", image.shape)
    #print("blob.shape:", blob.shape)

    # sets the blob as the input of the network
    net.setInput(blob)
    # get all the layer names
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # feed forward (inference) and get the network output
    # measure how much it took in seconds
    layer_outputs = net.forward(ln)

    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []
    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    count11=0
    result=[]
    # loop over the indexes we are keeping
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        print('x, y',x, y)
        #####################################################################
        ################ 4. LP is in targeted and 5 sec ##################
        ######################################################################
        if y >=376: # when LP is in targeted
            counts=counts+1
            print('counts',counts)
            if counts>=1: # 500 # when car over 5 sec
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                #text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                text = labels[class_ids[i]]
                #####################################################################
                ################ 5. get location of LP ##################
                ######################################################################
                lp_image = image[y:y + h,x:x + w]
                cv2.imshow('lp_image',lp_image)
                #print('text', text)
                #print('class_ids[i]', class_ids[i])
                if (class_ids[i] == 0):
                    counts = counts+1
                    print('counts', counts)
                    
                # calculate text width & height to draw the transparent boxes as background of the text
                (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                # add opacity (transparency to the box)
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                # now put the text (label: confidence %)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
                #cv2.imwrite(filename + "_yolo3." + ext, image)

                # convert to gray scale
                gray22 = cv2.cvtColor(lp_image, cv2.COLOR_BGR2GRAY)
                # blur to reduce noise
                gray22 = cv2.bilateralFilter(gray22, 11, 10, 10)
                
                ret, gray22 = cv2.threshold(gray22, 70, 255, cv2.THRESH_BINARY)
                cv2.imshow('gray22',gray22)

                contoursBLUnit=cv2.findContours(gray22.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                contoursBLUnit=imutils.grab_contours(contoursBLUnit)
                contoursBLUnit=sorted(contoursBLUnit,key=cv2.contourArea,reverse=True)[:1]

                for c in contoursBLUnit:
                    # approximate the contour
                    x1,y1,w1,h1 = cv2.boundingRect(c) 
                    #cv2.rectangle(lp_image,(x1,y1),(x1+w1,y1+h1),(0,255,0),2)

                    #####################################################################
                    ################ 6. get LP ##################
                    ######################################################################

                    crop_img_lp = lp_image[y1:y1+h1, x1:x1+w1] 
                    cv2.imshow('crop_img_lp',crop_img_lp)
                    cv2.imwrite('lpr_cap.jpg', crop_img_lp)

                    file3 = open('lpr_cap.jpg','rb').read()
                    # We must encode the file to get base64 string
                    file3 = base64.b64encode(file3) # save LP to base64


                    gray33 = cv2.cvtColor(crop_img_lp, cv2.COLOR_BGR2GRAY)
                    # blur to reduce noise
                    gray33 = cv2.bilateralFilter(gray33, 11, 10, 10)
                    ret, thresh2 = cv2.threshold(gray33, 120, 255, cv2.THRESH_BINARY_INV)
                    cv2.imwrite('lp_detected'+'.jpg',crop_img_lp)
                    cv2.imshow("thresh2", crop_img_lp)
                    #print('hs',hs)
                    #cv2.waitKey(0)
                    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
                    reader = easyocr.Reader(['th', 'en'], gpu=False)
                    print('reader',reader)
                    result = reader.readtext('lp_detected.jpg', detail=0)
                    print(result)
                    print(result[0])
                    ress = result[0] + result[1]
                    # fetching the dimensions
                    wid = crop_img_lp.shape[1]
                    hgt = crop_img_lp.shape[0]
                    # displaying the dimensions
                    print(str(wid) + "x" + str(hgt))
                    cnt_down=1
                        #################################################
                        ################ 11. Count Car ##################
                        #################################################
                        #cnt_down=1
                        #if ress == '66กฌ93':
                         #   ress='6กฌ93'

                        ###########################################
                        ########## SQL ############################
                        ###########################################
                        #print('len(ress)==5',len(ress))

                        #print('ch_rules',ch_rules)
                        #print('toc',toc)
                        
                    if wid>= 1000 and hgt >= 1000:
                        mycursor = mydb.cursor()
                        sql = "INSERT INTO lpr_name (tab_num, dates, times, city, number,count_car,text_tab,road,photo,photo_input) VALUES (%s,%s,%s, %s,%s, %s, %s, %s, %s, %s)"
                        val = (tab_num, str(dd)+'-'+str(mm)+'-'+str(yy), str(hh)+':'+str(minu)+':'+str(secc), "กรุงเทพมหานคร", ress,cnt_down,text_tab,road,file3,file2)
                        mycursor.execute(sql, val)
                        mydb.commit()
                        print(mycursor.rowcount, "record file3 inserted.")
                        



    cv2.imshow('frame',image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
