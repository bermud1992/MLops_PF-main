import joblib
import sklearn
from fastapi import FastAPI, HTTPException , APIRouter
import uvicorn
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
import mlflow

class CoverType(BaseModel):
    Elevation: List[float] = [0.0]
    Aspect: List[float] = [0.0]
    Slope: List[float] = [0.0]
    Horizontal_Distance_To_Hydrology: List[float] = [0.0]
    Vertical_Distance_To_Hydrology: List[float] = [0.0]
    Horizontal_Distance_To_Roadways: List[float] = [0.0]
    Hillshade_9am: List[float] = [0.0]
    Hillshade_Noon: List[float] = [0.0]
    Hillshade_3pm: List[float] = [0.0]
    Horizontal_Distance_To_Fire_Points: List[float] = [0.0]
    Wilderness_Area: List[str] = ['Rawah']
    Soil_Type: List[str] = ['C7745']
    Cover_Type: List[float] = [0.0]

app = FastAPI()

def decode_input(input):
    input_dict=dict(input)
    df = pd.DataFrame.from_dict(input_dict)
    model_columns = [
    'Elevation', 
    'Aspect', 
    'Slope', 
    'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
    'Wilderness_Area',
    'Soil_Type',
    'Cover_Type']
    df = df[model_columns]
    print(df)
    return df

@app.post("/predict/{model_name}")
def predict_model(input_data : CoverType,model_name: str = "modelo_base"):
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
    mlflow.set_tracking_uri("http://10.43.101.151:8087")
    mlflow.set_experiment("mlflow_tracking_examples")
    model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_production_uri)
    decoded_data = decode_input(input_data)
    prediction = loaded_model.predict(decoded_data)
    prediction_list = prediction.tolist()
    
    return {"model_used": model_name, "prediction":prediction_list}