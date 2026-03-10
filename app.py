import os
import pickle
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
 

# Pydantic Schemas
class PredictionRequest(BaseModel):
    """
    Input schema for diabetes classification.

    Attributes
    ----------
    age : float
    bmi : float
    bp : float
    s1 : float
    """
    age: float = Field(..., description="Age feature")
    bmi: float = Field(..., description="Body Mass Index feature")
    bp: float = Field(..., description="Average blood pressure feature")
    s1: float = Field(..., description="Total serum cholesterol feature")


class PredictionResponse(BaseModel):
    """
    Output schema for diabetes classification.

    Attributes
    ----------
    prediction : int
        1 = Has diabetes
        0 = No diabetes
    probability : float
        Probability of having diabetes
    """
    prediction: int
    probability: float


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    status: str


def train_model() -> None:
    """
    Train a logistic regression classifier using 4 features
    from sklearn diabetes dataset.

    Converts regression target into binary classification
    using median threshold.
    """
    diabetes = datasets.load_diabetes()

    # Select 4 features: age(0), bmi(2), bp(3), s1(4)
    X: np.ndarray = diabetes.data[:, [0, 2, 3, 4]]

    # Convert continuous target into binary
    y_continuous: np.ndarray = diabetes.target
    threshold: float = np.median(y_continuous)
    y: np.ndarray = (y_continuous > threshold).astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model

# Load model once at startup
model = train_model()


# FastAPI Application
app: FastAPI = FastAPI(
    title="Diabetes Classification API",
    description="Binary classification API for predicting diabetes.",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"message": "Welcome to the root route!"}

@app.get("/isAlive", response_model=HealthResponse)
def is_alive() -> HealthResponse:
    """
    Health check endpoint.
    """
    return HealthResponse(status="true")


@app.post("/prediction/", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict whether a patient has diabetes.

    Returns:
        prediction: 0 or 1
        probability: probability of class 1
    """
    features = np.array([[request.age, request.bmi, request.bp, request.s1]])

    prediction: int = int(model.predict(features)[0])
    probability: float = float(model.predict_proba(features)[0][1])

    return PredictionResponse(
        prediction=prediction,
        probability=probability
    )