import tensorflow as tf
import pandas as pd
import logging 
import time
import joblib

from typing import Literal
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field ,model_validator
from src.preprocessing.preprocess import preprocess_new_data
from fastapi.middleware.cors import CORSMiddleware
from src.explainability.shap_explainer import explain_prediction
from src.rag.retriver import generate_answer

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Insuranceinput(BaseModel):
    age:int= Field(
        # ... -->mean field req
        ...,
        gt=0 , le=100
    )
    sex : Literal["male" ,"female"]
    bmi: float=Field(
        ...,
        gt=0 , le=100
    )
    children: int=Field(
        ...,
        ge=0 , le=10
    )
    smoker : Literal["yes","no"]
    region :Literal[
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]

    @model_validator(mode="after")
    def validate_customer_data(self):

        if self.age < 18 and self.children > 0:
            raise ValueError(
                "Customer under 18 cannot have children"
            )

        return self
    
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
# @app.get("/")

# def home():
#     return{
#         "message":"API is running"
#     }

# @app.post("/predict")
# def predict(data:Insuranceinput):
#     return{
#         "message": "Input received successfully",
#         "data" : data
#     }



# load model whn appl start to avoid overhed
# model = tf.keras.models.load_model("src/model/insurance_model.keras")

# preprocessor = joblib.load("src/preprocessing/preprocessor.pkl")

MODEL_VERSION = "v1"

MODEL_PATH = (
    f"src/artifacts/{MODEL_VERSION}/insurance_model.keras"
)

PREPROCESSOR_PATH = (
    f"src/artifacts/{MODEL_VERSION}/preprocessor.pkl"
)

model = tf.keras.models.load_model(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

# # convert pydantic i/p -->dataframe: preprocess want same col struct -->panda preserve that
# @app.post("/predict")
# def predict(data:Insuranceinput):
#     input_data = pd.DataFrame([data.model_dump()])
#     print(input_data)

#     return{
#         "message":"Input converted successfully"
#     }

# add logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/predict")
def predict(data: Insuranceinput):
    try:
            logger.info("Prediction request received")

            total_start =time.perf_counter()

        # Convert Pydantic object to dictionary
            customer_data = data.model_dump()

            preprocessing_start =time.perf_counter()
            # Apply saved preprocessing
            X_new = preprocess_new_data(customer_data ,preprocessor)
            preprocessing_time =(
                 time.perf_counter() - preprocessing_start
            )* 1000

            # model inference
            inference_start = time.perf_counter()

            # Make prediction
            prediction = model.predict(X_new, verbose=0)
            inference_time =(
                 time.perf_counter() - inference_start
            ) *1000

            # Extract single prediction
            predicted_premium = float(prediction[0][0])
            # Generate SHAP explanation for this customer
            shap_explanation = explain_prediction(customer_data)

            total_time =(
                 time.perf_counter() -total_start
            )*1000

            logger.info(f"Preprocessing: {preprocessing_time:.2f} ms")

            logger.info(
            f"Model inference: {inference_time:.2f} ms")

            logger.info(
            f"Total request: {total_time:.2f} ms")

            return {
                "predicted_premium": predicted_premium,
                "explanation": shap_explanation,
            }

    except Exception:
         logger.exception("Prediction Failed")

         raise HTTPException(
              status_code=500,
              detail="Prediction failed"
         )

@app.post("/chat")
def chat(data: ChatRequest):
    try:
        logger.info("Chat request received")

        history = [
            {
                "role": message.role,
                "content": message.content
            }
            for message in data.history
        ]

        documents, answer = generate_answer(
            data.question,
            history
        )

        sources = []

        for document, score in documents:
            sources.append({
                "source": document.metadata.get("source"),
                "score": float(score)
            })

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Chat request failed"
        )