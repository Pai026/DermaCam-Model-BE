from ..db import product_collection
import tflite_runtime.interpreter as tflite
from PIL import Image
import urllib.request
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
model_file = "app/model/DiseaseDetectionModel.tflite"
severity_file = "app/model/SeverityModel.tflite"

def product_helper(product) -> dict:
    return {
        "product_name":product["product_name"],
        "brand":product["brand"],
        "brand_link":product["brand_link"],
        "ingredients": product["ingredients"],
        "image": product["image"],
        "cures": product["cures"]
    }

def get_user_disease_detail(imageUrl):
    detectionModel = tflite.Interpreter(model_file)
    resp = urllib.request.urlopen(imageUrl)
    input_details = detectionModel.get_input_details()
    detectionModel.resize_tensor_input(
    input_details[0]['index'], (1, 64, 64, 3))
    output_details = detectionModel.get_output_details()
    detectionModel.allocate_tensors()
    imageUploaded = np.asarray(bytearray(resp.read()), dtype="uint8")
    test_image = cv2.imdecode(imageUploaded, cv2.IMREAD_COLOR)
    resized_shape = (64,64)
    test_img = cv2.resize(test_image,(resized_shape[1],resized_shape[0]))
    skinDiseaseTypes=['blackhead', 'Acne', 'kutil filiform', 'flek hitam', 'folikulitis', 'milia', 'Dermatitis perioral', 'Karsinoma', 'panu', 'melanoma', 'herpes', 'Eksim', 'papula', 'whitehead', 'Tinea facialis', 'rosacea', 'Pustula', 'psoriasis']
    new_img = test_img.astype(np.float32)
    new_img /=255.0
    detectionModel.set_tensor(input_details[0]['index'], [new_img])
    detectionModel.invoke()
    output_data = detectionModel.get_tensor(output_details[0]['index'])
    ind=(np.argmax(output_data))
    if(skinDiseaseTypes[ind] == 'Acne'):
        severityModel = tflite.Interpreter(model_path=severity_file)
        severityModel.resize_tensor_input(
        input_details[0]['index'], (1, 64, 64, 3))
        output_details = severityModel.get_output_details()
        severityModel.allocate_tensors()
        severityLevel=['Level_0', 'Level_1','Level_2']
        severityModel.set_tensor(input_details[0]['index'], [new_img])
        severityModel.invoke()
        output_data = severityModel.get_tensor(output_details[0]['index'])
        index=(np.argmax(output_data))
        if(severityLevel[index]=="Level_0"):
            products=product_collection.find({'cures':skinDiseaseTypes[ind]})
            result=[]
            for i in products:
                result.append(product_helper(i))
            return {
                "Prediction":skinDiseaseTypes[ind],
                "SeverityLevel":severityLevel[index],
                "Suggested_Products":result
            }
        else:
            return {
            "Prediction":skinDiseaseTypes[ind],
            "SeverityLevel":severityLevel[index],
            "Suggestion":"Please consult nearby doctors"
            }
    return {
        "Prediction":skinDiseaseTypes[ind],
        "Suggestion":"Please consult nearby doctors"
        }