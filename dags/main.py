from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import joblib
import numpy as np
import mlflow
import os
import mysql.connector
import pandas as pd
import requests
from io import StringIO
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC


def write_to_raw_data():
    # Download data
    link = "https://docs.google.com/uc?export=download&confirm={{VALUE}}&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC"
    response = requests.get(link)
    csv_content = response.content.decode('utf-8')
    csv_file = StringIO(csv_content)
    df = pd.read_csv(csv_file)
    # Split the data into train_val and test sets (70% and 30%)
    train, val_test = train_test_split(df, test_size=0.3, random_state=42)

    # Split the val_test set into validation and test sets (15% and 15%)
    validation, test = train_test_split(val_test, test_size=0.5, random_state=42)

    # Add a 'type' column to identify the data split
    train['type'] = 'train'
    validation['type'] = 'validation'
    test['type'] = 'test'
    # Conection to MySql 
    conn = mysql.connector.connect(
        host="mysql",
        user="airflow",
        password="airflow",
        database="airflow"
    )
    cursor = conn.cursor()
    engine = create_engine("mysql+mysqlconnector://airflow:airflow@mysql/airflow")

    # To sql
    train.to_sql('raw_data', con=engine, if_exists='replace', index=False, chunksize = 15000)

    # Confirm and close
    conn.commit()
    conn.close()

def write_to_clean_data():
    # Conexion a la bd
    conn = mysql.connector.connect(
        host="mysql",
        user="airflow",
        password="airflow",
        database="airflow"
    )

    cursor = conn.cursor()
    engine = create_engine("mysql+mysqlconnector://airflow:airflow@mysql/airflow")

    # Ejecuta sql e inserta en un df
    query = "SELECT * FROM raw_data WHERE type = 'train';"
    df = pd.read_sql_query(query, conn)

    # Representar adecuadamente los nan
    df = df.replace("?",np.nan)
    # Se dropean columnas con nan
    df.drop(['weight','payer_code','medical_specialty'], axis=1, inplace=True)
    # Dropear observaciones 'Unknown/Invalid' 
    invalid_rows = df[df.isin(['Unknown/Invalid'])].any(axis=1)
    df = df[~invalid_rows].copy()    
    # Se dropean los duplicados por paciente y se deja el primero
    df.drop_duplicates(subset ="patient_nbr", keep = "first", inplace = True)
    # Dropear columnas con valores unicos = 1
    df.drop(['examide','citoglipton','glimepiride-pioglitazone','max_glu_serum','A1Cresult'], axis='columns', inplace=True)

    # Conection to MySql 
    df.to_sql('clean_data', con=engine, if_exists='replace', index=False)

    # Confirm and close
    conn.commit()
    conn.close()

def train_models():

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
    mlflow.set_tracking_uri("http://10.43.101.151:8087")
    mlflow.set_experiment("mlflow_tracking_examples")

    # Conexion a la bd
    conn = mysql.connector.connect(
        host="mysql",
        user="airflow",
        password="airflow",
        database="airflow"
    )

    cursor = conn.cursor()
    engine = create_engine("mysql+mysqlconnector://airflow:airflow@mysql/airflow")

    query = "SELECT * FROM clean_data"
    df = pd.read_sql_query(query, conn)
    conn.commit()
    conn.close()

    categorical_features = ['race', 'gender', 'age', 'admission_type_id','discharge_disposition_id', 
                            'admission_source_id','metformin', 'repaglinide', 
                            'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
                            'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                            'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 
                            'glipizide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone',
                            'change', 'diabetesMed', 'diag_1', 'diag_2','diag_3'
                        ]

    numerical_features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
                        'number_outpatient', 'number_emergency','number_inpatient', 'number_diagnoses']

    target = "readmitted"
    all_features = categorical_features + numerical_features

    X_train = df[all_features]

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(df[target]).ravel()

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])

    models = {
        'RandomForestClassifier': RandomForestClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
    }

    param_grids = {
        'RandomForestClassifier': {
            'max_depth': [1, 2, 3, 10],
            'n_estimators': [10, 11]
        },
        'DecisionTreeClassifier': {
            'max_depth': [None, 1, 2, 3]
        }
    }

    for model_name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        param_grid = {'classifier__' + key: value for key, value in param_grids[model_name].items()}
        
        search = GridSearchCV(pipe, param_grid, n_jobs=-3)

        with mlflow.start_run(run_name=f"autolog_pipe_{model_name}") as run:
            search.fit(X_train, y_train)
            mlflow.log_params(param_grid)
            mlflow.log_metric("best_cv_score", search.best_score_)
            mlflow.log_params("best_params", search.best_params_)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,    
    'start_date': datetime(2024, 5, 3),
    'retries': 1,
}

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023,3,4),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG('main', default_args=default_args, schedule_interval='@once') as dag:
    write_to_raw_data_task = PythonOperator(
        task_id='write_to_raw_data_task',
        python_callable=write_to_raw_data
    )
    write_to_clean_data_task = PythonOperator(
        task_id='write_to_clean_data_task',
        python_callable=write_to_clean_data
    )
    train_models_task = PythonOperator(
        task_id='train_models_task',
        python_callable=train_models
    )
write_to_raw_data_task >> write_to_clean_data_task >> train_models_task
