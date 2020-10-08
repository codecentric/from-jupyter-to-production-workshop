from typing import Optional

from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


class QuerySum(BaseModel):
    a: int
    b: int

class Result(BaseModel):
    c: int
        
@app.get("/")
def hello_world():
    return {"Hello": "World"}

@app.post("/add", response_model=Result)
def hello_world(query_sum: QuerySum):
    return Result(c=query_sum.a + query_sum.b)
