from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
import io
from pymongo import MongoClient

# Подключение к MongoDB
mongo_client = MongoClient("mongodb://mongo:mongo@localhost:27017/mongo")
db = mongo_client["animal_classification"]
result_collection = db["prediction_results"]


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Загрузка модели
model_path = "./animalsv2.keras"
model = load_model(model_path)

# Список видов животных
translate = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

class PredictionResult(BaseModel):
    label: str
    selected_animal: str
    correct: bool

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...), selected_animal: str = Form(...)):
    content = await file.read()
    img = image.load_img(io.BytesIO(content), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_label = translate[prediction.argmax()]
    correct = selected_animal.lower() == predicted_label.lower()

    # Сохранение результата в MongoDB
    result_collection.insert_one({
        "selected_animal": selected_animal,
        "predicted_animal": predicted_label,
        "correct": correct,
        "timestamp": datetime.now()
    })

    # Запрос для статистики
    total_count = result_collection.count_documents({})
    correct_count = result_collection.count_documents({"correct": True})

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": {"label": predicted_label, "selected_animal": selected_animal, "correct": correct},
            "total_count": total_count,
            "correct_count": correct_count
        }
    )
