from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class QuerySum(BaseModel):
    a: int
    b: int


class Result(BaseModel):
    c: int


@app.get("/")
def hello_world():
    return {"Hello": "World"}


@app.post("/add")
def add(query_sum: QuerySum) -> Result:
    return Result(c=query_sum.a + query_sum.b)