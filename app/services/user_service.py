from ..db import user_collection,product_collection
from .auth_service import AuthHandler
from fastapi.encoders import jsonable_encoder
from fastapi import HTTPException
from bson.objectid import ObjectId
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import urllib.request
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
model_file = "app/model/DiseaseDetectionModel.h5"
severity_file = "app/model/SeverityModel.h5"

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
    detectionModel = load_model(model_file)
    resp = urllib.request.urlopen(imageUrl)
    imageUploaded = np.asarray(bytearray(resp.read()), dtype="uint8")
    test_image = cv2.imdecode(imageUploaded, cv2.IMREAD_COLOR)
    resized_shape = (64,64)
    test_img = cv2.resize(test_image,(resized_shape[1],resized_shape[0]))
    skinDiseaseTypes=['blackhead', 'Acne', 'kutil filiform', 'flek hitam', 'folikulitis', 'milia', 'Dermatitis perioral', 'Karsinoma', 'panu', 'melanoma', 'herpes', 'Eksim', 'papula', 'whitehead', 'Tinea facialis', 'rosacea', 'Pustula', 'psoriasis']
    x = image.img_to_array(test_img)
    x = np.expand_dims(x, axis = 0)

    x /= 255

    custom = detectionModel.predict(x)
    x = np.array(x, 'float32')
    a=custom[0]
    ind=np.argmax(a)
    if(skinDiseaseTypes[ind] == 'Acne'):
        severityModel = load_model(severity_file)
        severityLevel=['Level_0', 'Level_2', 'Level_1']
        custom = severityModel.predict(x)
        x = np.array(x, 'float32')
        a=custom[0]
        index=np.argmax(a)
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