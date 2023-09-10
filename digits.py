import cv2
import numpy as np
import time
import os
import datetime
from datetime import datetime
from PIL import Image
from io import BytesIO
import requests
from scipy import ndimage

import random
import string




INPUT_WIDTH = 320
INPUT_HEIGHT = 320
SCORE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
classesFile = "coco.names"
classes = None


ch_detection_modelWeights = "best_upwork.onnx"
ch_detection_model = cv2.dnn.readNet(ch_detection_modelWeights)
x_=[".","0","1","2","3","4","5","6","7","8","9"]






def pre_process(input_image, net,w,h):
      # Create a 4D blob from a frame.
      #print(input_image.shape)
      blob = cv2.dnn.blobFromImage(input_image, 1/255,  (w, h), [0,0,0], 1, crop=False)

      # Sets the input to the network.
      net.setInput(blob)

      # Run the forward pass to get output of the output layers.
      outputs = net.forward(net.getUnconnectedOutLayersNames())
      return outputs

def get_xyxy(input_image,image_height,image_width, outputs,w,h):
      # Lists to hold respective values while unwrapping.
      class_ids = []
      confidences = []
      boxes = []
      output_boxes=[]
      results_cls_id=[]
      # Rows.
      rows = outputs[0].shape[1]

      x_factor = image_width / w
      y_factor =  image_height / h
      # Iterate through detections.
      for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= CONFIDENCE_THRESHOLD:
                  classes_scores = row[5:]
                  # Get the index of max class score.
                  class_id = np.argmax(classes_scores)
                  #  Continue if the class score is above threshold.
                  if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height,])
                        boxes.append(box)
 # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
      indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
      for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3] 
            results_cls_id.append(class_ids[i])
         
            # cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 1)

            boxes[i][2]=left + width
            boxes[i][3]=top + height
            #check if the height is suitable
            output_boxes.append(boxes[i])
      # cv2.imwrite('x1.jpg',input_image)
      return output_boxes,results_cls_id #boxes (left,top,width,height)

def char_det(input_image,ch_detection_model,w,h):
      #in_image_copy=input_image.copy()
      detections = pre_process(input_image.copy(), ch_detection_model,w,h) #detection results 
      image_height=input_image.shape[0]
      image_width=input_image.shape[1]
      bounding_boxes=get_xyxy(input_image,image_height,image_width, detections,w,h)

      return bounding_boxes

def rearange_(array_pred,results_cls_id):
      scores=''
      #print(y2,y2[:,0])
      ind=np.argsort(array_pred[:,0])

      #print(license_image.shape[0],ind)
      for indx in (ind):
        scores=scores+x_[results_cls_id[indx]]
        

      return scores


def main_func(img):
      results={"num":"","time":"0","info":"0"}
      scores=0
      t1=time.time()
      img = np.array(img) 
      im2=img.copy()

      width_height_diff=img.shape[1]-img.shape[0] #padding

      if width_height_diff>0:
            img = cv2.copyMakeBorder(img, 0, width_height_diff, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
      if width_height_diff<0:
            img = cv2.copyMakeBorder(img, 0, 0, 0, int(-1*width_height_diff), cv2.BORDER_CONSTANT, (0,0,0))

    
      cropped_chars_array,results_cls_id=char_det(img.copy(),ch_detection_model,320,320)
      if len(cropped_chars_array)!=0:
            cropped_chars_array=np.asarray(cropped_chars_array)
            scores=rearange_(cropped_chars_array,results_cls_id)
            results["num"]=scores
            results["info"]="200"
      
      time_of_process=(time.time()-t1)
      results["time"]=str(time_of_process)
      
      
      #return scores,time_of_process
      return results
