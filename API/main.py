import mlflow.pyfunc
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import pandas as pd
from io import StringIO

import global_

model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model["save_model"] = mlflow.pyfunc.load_model(global_.paths.path_to_save_model)
    yield
    model.clear()

app = FastAPI(lifespan=lifespan)


@app.post("/set_flower_params")
async def set_params(data: Request):
    input_data = await data.json()
    iris_df = pd.read_json(StringIO(input_data))
    s_model = model["save_model"]
    result = s_model.predict(iris_df)
    return result.to_json()


@app.get("/get_input_column")
def get_input_col():
    return model["save_model"].metadata.signature.inputs

@app.get("/get_output_column")
def get_out_col():
    return model["save_model"].metadata.signature.outputs
