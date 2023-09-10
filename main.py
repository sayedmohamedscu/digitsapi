from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image, ImageOps
import numpy as np
import cv2
import io, json
import base64
import digits
import base64
import img_checker
import time

app = FastAPI()

# load a pre-trained Model and convert it to eval mode. 
# This model loads just once when we start the API.
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

#model.eval()

# define the Input class
class Input(BaseModel):
	base64str : str
	UID : str

def base64str_to_PILImage(base64str):
    is_valid_base64=1
    img_=1
    try:    
        base64_img_bytes = base64str.encode('utf-8')
        base64bytes = base64.b64decode(base64_img_bytes)
        bytesObj = io.BytesIO(base64bytes)
        img_=ImageOps.exif_transpose(Image.open(bytesObj))
    except :
        is_valid_base64=0
    return img_,is_valid_base64



@app.put("/getdigit")
async def get_predictionbase64(d:Input):

	results={"num":"","time":"0","info":"0"}
	time_of_process=0
	img,is_valid_base64_image = base64str_to_PILImage(d.base64str)
	if is_valid_base64_image ==1: #before starting the process
		img = np.array(img)

		converted_img,checker_results=img_checker.img_checker_pipeline(img)
		

		if checker_results['is_img']==1 and checker_results['normal_img_shape']==1 and checker_results['img_channels']!='not_rgb_or_rgbA':
			converted_img = converted_img[:, :, ::-1].copy()
			#converted_img=QR_helper.resizing_without_distortion_module(converted_img)
			# converted_img=cv2.resize(converted_img,(512,512))
			results=digits.main_func(converted_img)
		#results=np.asarray(results)
		



	return results
