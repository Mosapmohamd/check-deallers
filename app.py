# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Scammer Detection API")

classifier = pipeline("zero-shot-classification")

class TextInput(BaseModel):
    text: str

@app.post("/check-scammer")
def check_scammer(input: TextInput):
    result = classifier(
        input.text,
        candidate_labels=["Real", "Dealer"]
    )

    is_real = result["labels"][0].strip() == "Real"

    return {
        "is_real": is_real,
        "top_label": result["labels"][0],
        "scores": dict(zip(result["labels"], result["scores"]))
    }
