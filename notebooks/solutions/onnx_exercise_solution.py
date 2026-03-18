import numpy as np
import onnxruntime as rt
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

# create the fastapi application
api = FastAPI()

# load the onnx model. for our demo purposes, the model may live in the global context.
# for more advanced apis, it may be reasonable to move the model into a dependency.
# HINT: The onnx model is supposed to be served by an InferenceSession. Check out the
# onnx_introduction.ipynb to find a use of this object.
sess = rt.InferenceSession("model/loan_model.onnx")


# define the features that our request expects
# for this task, have a look at the onnx_introduction.ipynb to find the feature names.
# remember that each feature needs to be annotated with its respective type.
# in this task, it is either a float or a str feature
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
        """Utility function to get all features of this BaseModel of a certain type.

        Example usage:
        `all_int_fields = LoanFeatures.get_type_fields(int)`  # result: {}
        """
        return {
            field for field, fieldinfo in cls.model_fields.items()
            if fieldinfo.annotation == _type
        }




# create a POST endpoint that retrieves prediction requests, performs the prediction and
# returns the prediction result as its response
@api.post("/predict")
def post_predict(loan_features: LoanFeatures) -> Literal["Y", "N"]:
    # convert the request object into a dict of numpy arrays
    ins = {k: np.array(v) for k, v in loan_features.model_dump().items()}

    # convert the feature dtypes into the same dtypes that the model expects
    # NOTE that the input_value is expected to be two-dimensional, so we need to reshape it properly
    for c in LoanFeatures.get_type_fields(float):
        ins[c] = ins[c].astype(np.float32).reshape((1, 1))
    for k in LoanFeatures.get_type_fields(str):
        ins[k] = ins[k].reshape((1, 1))

    # run the prediction
    labels, _ = sess.run(None, ins)

    # create the response.
    # HINT: The type hint of this function's return value shows that we want to return
    # either "Y" or "N". Have a look if you can find this in the response of the
    # inference session.
    return labels[0]



# define the features that our response should send
class Prediction(BaseModel):
    label: Literal["Y", "N"]

# create a POST endpoint that retrieves prediction requests, performs the prediction and
# returns the prediction result as its response
@api.post("/predict_with_response_model")
def post_predict_with_response_model(loan_features: LoanFeatures) -> Prediction:
    # convert the request object into a dict of numpy arrays
    ins = {k: np.array(v) for k, v in loan_features.model_dump().items()}

    # convert the feature dtypes into the same dtypes that the model expects
    # NOTE that the input_value is expected to be two-dimensional, so we need to reshape it properly
    for c in LoanFeatures.get_type_fields(float):
        ins[c] = ins[c].astype(np.float32).reshape((1, 1))
    for k in LoanFeatures.get_type_fields(str):
        ins[k] = ins[k].reshape((1, 1))

    # run the prediction
    labels, _ = sess.run(None, ins)

    # create the response object.
    return Prediction(label=labels[0])
