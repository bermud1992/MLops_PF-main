import joblib
import sklearn
from fastapi import FastAPI, HTTPException , APIRouter
import uvicorn
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
import mlflow

class PropertyDetails(BaseModel):
    brokered_by: List[float] = [8103.0]
    status: List[str] = ['sold']
    price: List[float] = [375900.0]
    bed: List[float] = [3.0]
    bath: List[float] = [1.0]
    acre_lot: List[float] = [1.2]
    street: List[float] = [1467938.0]
    city: List[str] = ['Kennett Square']
    state: List[str] = ['Pennsylvania']
    zip_code: List[float] = [19348.0]
    house_size: List[float] = [1995.0]
    prev_sold_date: List[str] = ['2022-01-21']
    batch_number: List[float] = [4]
    max_batch_number: List[float] = [4]
    last_batch_number: List[float] = [4]

app = FastAPI()

def decode_input(input):
    input_dict = dict(input)
    df = pd.DataFrame.from_dict(input_dict)
    model_columns = [
        'brokered_by', 
        'status', 
        'price', 
        'bed', 
        'bath', 
        'acre_lot', 
        'street', 
        'city', 
        'state', 
        'zip_code', 
        'house_size', 
        'prev_sold_date', 
        'batch_number', 
        'max_batch_number', 
        'last_batch_number'
    ]
    df = df[model_columns]
    print(df)
    return df

@app.post("/predict/{model_name}")
def predict_model(input_data : PropertyDetails,model_name: str = "modelo_base"):
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
    mlflow.set_tracking_uri("mlflow:5000")
    mlflow.set_experiment("mlflow_tracking_examples")
    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)
    decoded_data = decode_input(input_data)
    prediction = loaded_model.predict(decoded_data)
    prediction_list = prediction.tolist()
    
    return {"model_used": model_name, "prediction":prediction_list}