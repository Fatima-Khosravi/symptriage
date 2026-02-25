import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Input(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "SympTriage API is running"}

@app.post("/predict")
def predict(input: Input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Extract symptoms and suggest disease from: {input.text}"}
        ]
    )


    return {"result": response.choices[0].message.content}
