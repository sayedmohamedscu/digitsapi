from PIL import Image, ImageOps
import numpy as np
import cv2
import io, json
import base64

def is_img_checker(input_img):
    try :
        img_shape=input_img.shape
        return True
    except :
        return False

def Dimensions_checker(input_img):
    if len (input_img.shape)==3: #it's a 3d image x,y,chs
        c=1 #dummy data
        return True,input_img
    elif len (input_img.shape)==2:
        if input_img.max()<2:
            input_img=input_img*255 #if img values is normalized to 0,1
        input_img = cv2.cvtColor(input_img,cv2.COLOR_GRAY2RGB)
        return True,input_img
    else :
        c=0 #not an ordinary image
        return False,input_img

def channels_checker(input_img):
    if (input_img.shape[2])==3: #jpg
        c=1 #dummy data
        return '3chs'
    elif (input_img.shape[2])==4: #png
        c=2 #dummy data
        return '4chs'
    else:
        return 'not_rgb_or_rgbA'

def img_checker_pipeline(input_img):
    check_dict={'is_img':0,'normal_img_shape':0,'img_channels':'not_rgb_or_rgbA'}
    if is_img_checker(input_img):
        check_dict['is_img']=1
        statues,input_img= Dimensions_checker(input_img)
        if statues:#if true
            check_dict['normal_img_shape']=1
            result_ch_check = channels_checker (input_img)
            if result_ch_check=='3chs':
                check_dict['img_channels']=3
            if result_ch_check=='4chs':
                check_dict['img_channels']=4 
                input_img=input_img[:,:,:3]
            if result_ch_check=='not_rgb_or_rgbA':
                check_dict['img_channels']='not_rgb_or_rgbA'
     
    return input_img,check_dict     
