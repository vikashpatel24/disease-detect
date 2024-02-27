from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
from starlette.responses import HTMLResponse
from starlette.requests import Request
from fastapi.templating import Jinja2Templates

# Load TFLite model
model_path = './utils/disease_detect_model.tflite'
tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
tflite_interpreter.allocate_tensors()

# Assuming disease_dic is imported from utils.diseases
from utils.diseases import disease_dic

# Disease names
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

app = FastAPI()

# Enable CORS for all routes
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "message": "Welcome to Disease Detection API!"})

@app.post("/disease-detect")
async def disease_prediction(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        image = read_file_as_image(file_content)
        image = tf.image.resize(image, [256, 256]).numpy()
        image = np.expand_dims(image, 0)

        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        tflite_interpreter.allocate_tensors()

        tflite_interpreter.set_tensor(input_details[0]['index'], image)
        tflite_interpreter.invoke()
        predictions = tflite_interpreter.get_tensor(output_details[0]['index'])

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = round(np.max(predictions[0]) * 100, 2)

        # Log image details
        image_name = file.filename
        image_size = len(file_content)
        print(f"Image Name: {image_name}, Image Size: {image_size} bytes")

        data = {
            'confidence': float(confidence),
            'data': str(disease_dic[predicted_class])
        }
        print(confidence, predicted_class)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
