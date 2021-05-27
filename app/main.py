from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .services import user_service
from .db import cloudinary

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def read_root():
    return {"Message": "Welcome to DermaCam Model Backend End !!!"}


@app.post("/uploadImage", response_description="Return The disease to be detected")
async def uploadImage(image: UploadFile = File(...)):
    result = cloudinary.uploader.upload(image.file)
    return user_service.get_user_disease_detail(result['url'])
