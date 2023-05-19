import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn

class DataType(BaseModel):
    resp: int
    # prSeq: str
    # rtSeq: str
    vlT0: float
    # cd4T0: int


"""
{
  "VL_t0": 4.3,
  "CD4_t0": 145
}

"""

app = FastAPI()

with open("hiv_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
async def predict(item: DataType):
    df = pd.DataFrame([item.dict()])
    finalmod = model.predict(df)
    if finalmod[0] == 0:
        return {"prediction": "HIV Negative"}
    else:
        return {"prediction": "HIV Positive"}

@app.get("/")
async def root():
    return {"message": "This API Only Has Get Method as of now"}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)

    # app.run(app, host='0.0.0.0', port=8000)
