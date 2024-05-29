import joblib
import sklearn
from fastapi import FastAPI, HTTPException , APIRouter
from mlflow.client import MlflowClient
import uvicorn
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
import mlflow

class PropertyDetails(BaseModel):
    brokered_by: List[float] = [8103.0]
    status: List[str] = ['sold']
    city: List[str] = ['Kennett Square']
    state: List[str] = ['Pennsylvania']
    zip_code: List[float] = [19348.0]
    bed: List[float] = [3.0]
    bath: List[float] = [1.0]
    acre_lot: List[float] = [1.2]
    house_size: List[float] = [1995.0]
    price: List[float] = [375900.0]
    
app = FastAPI()

def decode_input(input):
    input_dict = dict(input)
    df = pd.DataFrame.from_dict(input_dict)
    model_columns = [
        'brokered_by', 
        'status', 
        'city', 
        'state', 
        'zip_code', 
        'bed', 
        'bath', 
        'acre_lot', 
        'house_size',
        'price'  
    ]
    df = df[model_columns]
    print(df)
    return df

def load_model_with_tag(experiment_name, tag_key, tag_value):
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = client.search_runs([experiment.experiment_id], filter_string=f"tags.{tag_key} = '{tag_value}'")
    if not runs:
        raise Exception(f"No se encontró ningún modelo con el tag {tag_key} = {tag_value}")
    # Asumimos que solo hay un modelo con este tag, tomar el último run
    last_run = runs[0]
    model_uri = f"runs:/{last_run.info.run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

@app.post("/predict/{model_name}")
def predict_model(input_data : PropertyDetails,model_name: str = "modelo_base"):
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
    mlflow.set_tracking_uri("http://mlflow:5000")
    #mlflow.set_experiment("mlflow_tracking_examples")
    model_production_uri = "models:/{model_name}@production".format(model_name=model_name)
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)
    #loaded_model = load_model_with_tag("mlflow_tracking_examples", "stage", "production") 
    decoded_data = decode_input(input_data)
    prediction = loaded_model.predict(decoded_data)
    prediction_list = prediction.tolist()
    
    return {"model_used": model_name, "prediction":prediction_list}