from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Schema for Text Data
class TextData(BaseModel):
    texts: List[str]

# Root path
@app.get("/")
def root():
    return {"message": "Análisis de Sentimiento API activa"}

# Ruta para analizar sentimientos (POST es mejor para enviar datos)
@app.post("/analyze/")
def analyze_sentiment(data: TextData):
    # 'data.texts' vendrá del cuerpo del JSON que envíes
    results = sentiment_pipeline(data.texts)
    return {"results": results}
