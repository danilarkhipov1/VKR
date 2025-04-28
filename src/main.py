from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from models import WasteClassifier
from utils import load_and_preprocess_image, format_prediction_result


app = FastAPI(title="Waste Classification API",
              description="API для классификации мусорных баков",
              version="1.0.0")

MODEL_PATH = "models/model.pt"
classifier = WasteClassifier(MODEL_PATH)

@app.get("/")
async def root():
    return {"message": "Сервис классификации мусорных баков"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Чтение и предобработка изображения
        image_data = await file.read()
        image = load_and_preprocess_image(image_data)
        
        # Получаем предсказания
        predicted_labels, predicted_probs = classifier.predict(image)
        
        # Форматируем результат
        result = format_prediction_result(predicted_labels, predicted_probs)
        
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )