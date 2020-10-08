import uvicorn
from fastapi import FastAPI
from fastapi import APIRouter


api = APIRouter()
from pydantic import BaseModel

import numpy as np
import onnxruntime as rt

sess = rt.InferenceSession("model/loan_model.onnx")

numeric_features = ['applicantincome', 'coapplicantincome', 'loanamount',
       'loan_amount_term', 'credit_history']
categorical_features = ['married', 'dependents', 'education',
       'self_employed', 'property_area']

class LoanFeatures(BaseModel):
    married: str
    dependents: str
    education: str
    self_employed: str
    property_area: str
    applicantincome: float
    coapplicantincome: float
    loanamount: float
    loan_amount_term: float
    credit_history: float

class Prediction(BaseModel):
    label: str
    probability: float

class PredictionResult(BaseModel):
    predicted: list

def predict(features: LoanFeatures) -> PredictionResult:
    ins = { k: np.array(v) for k,v in features.dict().items() } 
    for c in numeric_features:
        ins[c] = ins[c].astype(np.float32).reshape((1, 1))
    for k in categorical_features:
        ins[k] = ins[k].astype(object).reshape((1, 1)).astype(object)
    labels, probabilities = sess.run(None, ins)  
    predicted = []  
    for k, v in probabilities[0].items():
        predicted.append(Prediction(label=k, probability=v))
    return PredictionResult(predicted=predicted)

@api.post("/predict", response_model=PredictionResult)
def post_predict(
    loan_features: LoanFeatures,
):
    return predict(loan_features)

app = FastAPI()

app.include_router(api)