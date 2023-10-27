import numpy as np
import onnxruntime as rt
from fastapi import FastAPI
from pydantic import BaseModel

api = FastAPI()

sess = rt.InferenceSession("model/loan_model.onnx")


class LoanFeatures(BaseModel):
    applicantincome: float
    coapplicantincome: float
    loanamount: float
    loan_amount_term: float
    credit_history: float
    married: str
    dependents: str
    education: str
    self_employed: str
    property_area: str

    @classmethod
    def get_type_fields(cls, _type) -> set:
        return {
            field for field, fieldinfo in cls.model_fields.items()
            if fieldinfo.annotation == _type
        }


class Prediction(BaseModel):
    label: int
    probability: float


@api.post("/predict")
def post_predict(loan_features: LoanFeatures) -> list[Prediction]:
    ins = {k: np.array(v) for k, v in loan_features.model_dump().items()}

    for c in LoanFeatures.get_type_fields(float):
        ins[c] = ins[c].astype(np.float32).reshape((1, 1))
    for k in LoanFeatures.get_type_fields(str):
        ins[k] = ins[k].astype(object).reshape((1, 1)).astype(object)

    labels, probabilities = sess.run(None, ins)
    predicted = []
    for k, v in probabilities[0].items():
        predicted.append(Prediction(label=k, probability=v))

    return predicted