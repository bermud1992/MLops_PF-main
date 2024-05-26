import os
import mlflow
import requests
import numpy as np
import pandas as pd
import mysql.connector
import joblib
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow_name = f"proyecto2model"
# mlflow.set_experiment(mlflow_name)
# desc = "modelo del proyecto 2"
# with mlflow.start_run(run_name="run1", description=desc) as run:
#     mlflow.autolog(log_model_signatures=True, log_input_examples=True)
#     model_info = mlflow.sklearn.log_model(
#         sk_model=,
#         artifact_path=mlflow_name,
#         input_example=,
#         registered_model_name=mlflow_name,
#     )
#     print('tracking uri:', mlflow.get_tracking_uri())
#     print('artifact uri:', mlflow.get_artifact_uri())
#     print("model_uri", model_info.model_uri)

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
mlflow.set_tracking_uri("http://10.43.101.151:8087")
mlflow.set_experiment("mlflow_tracking_examples")

def get_data():
    conn = mysql.connector.connect(
    host="mysql",
    user="airflow",
    password="airflow",
    database="airflow"
    )
    select_query = """
    SELECT
    *
    FROM
        data_table
    """
    df = pd.read_sql(select_query, con=conn)
    conn.close()

    columns_to_convert = df.columns.difference(['Wilderness_Area', 'Soil_Type'])
    df[columns_to_convert] = df[columns_to_convert].astype(float)
    df = df.drop_duplicates()
    return df


def train_model():
    df =  ()
    # Set the target values
    y = df['Cover_Type']#.values
    features = list(df.columns[:-1])
    # Set the input values
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                            ["Wilderness_Area", "Soil_Type"]),
                                            remainder='passthrough') # pass all the numeric values through the pipeline without any changes.
    pipe = Pipeline(steps=[("column_trans", column_trans),("scaler", StandardScaler(with_mean=False)), ("RandomForestClassifier", RandomForestClassifier())])
    
    param_grid =  {'RandomForestClassifier__max_depth': [1,2,3,10], 'RandomForestClassifier__n_estimators': [10,11]}
    search = GridSearchCV(pipe, param_grid, n_jobs=2)
    
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True, registered_model_name="modelo_base")
    with mlflow.start_run(run_name="autolog_pipe_model_reg") as run:
        search.fit(X_train, y_train)
    
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023,3,4),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

schedule_interval = "*/50 * * * *"

dag = DAG(
    'Model_train_dag',
    default_args=default_args,
    description='Train model every 5 minutes',
    schedule_interval=schedule_interval,
    catchup=False
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    dag=dag,
)

train_model_task
