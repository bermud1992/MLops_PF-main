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
import shap
from sqlalchemy import create_engine, inspect
import logging
from category_encoders import TargetEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import ks_2samp
from sqlalchemy import create_engine, inspect


def write_to_raw_data():
    # Download data
    COLUMN_NAMES = ["brokered_by",
                    "status",
                    "price",
                    "bed",
                    "bath",
                    "acre_lot",
                    "street",
                    "city",
                    "state",
                    "zip_code",
                    "house_size",
                    "prev_sold_date"]
    
        # Configuración del logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    url = "http://10.43.101.149/data"
    params = {'group_number': '2'}
    headers = {'accept': 'application/json'}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        json_data = response.json()
        df = pd.DataFrame.from_dict(json_data["data"])
        df.columns = COLUMN_NAMES
        df["batch_number"] = json_data["batch_number"]   

        # Connect to MySQL and create table if not exists
        engine = create_engine("mysql+mysqlconnector://airflow:airflow@mysql/airflow")
        with engine.connect() as conn:
            table_exists = engine.dialect.has_table(conn, 'raw_data')
            if not table_exists:
                print("No existe la tabla")
                df.iloc[:0].to_sql('raw_data', con=engine, if_exists='replace', index=False)
            # Merge data into the table
            df.to_sql('raw_data', con=engine, if_exists='append', index=False, chunksize=10000)

    else:
        logger.error("Error al realizar la solicitud: %d", response.status_code)


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

    # Ejecuta sql
    query = """
            WITH max_batch AS (
                SELECT *,
                MAX(batch_number) OVER () AS max_batch_number
                FROM raw_data
            )
            SELECT
            *
            FROM max_batch
            WHERE batch_number = max_batch_number        
            ;
            """
    df = pd.read_sql_query(query, conn)

    CATEGORICAL_FEATURES = ["brokered_by",
                            "status",
                            "street",
                            "city",
                            "state",
                            "zip_code",
                            "prev_sold_date"]

    NUMERICAL_FEATURES = ["price",
                        "bed",
                        "bath",
                        "acre_lot",
                        "house_size"]
    
    # Impute missing values for categorical features with the mode
    for feature in CATEGORICAL_FEATURES:
        mode_value = df[feature].mode()[0]
        df[feature].fillna(mode_value, inplace=True)

    # Impute missing values for numerical features with the median
    for feature in NUMERICAL_FEATURES:
        median_value = df[feature].median()
        df[feature].fillna(median_value, inplace=True)
        
    # Record the initial number of rows
    initial_rows = df.shape[0]

    # Remove values outside the 0.25th and 99.95th percentiles
    for feature in NUMERICAL_FEATURES:
        lower_bound = df[feature].quantile(0.0025)
        upper_bound = df[feature].quantile(0.9995)
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

    # Record the number of rows after filtering
    final_rows = df.shape[0]

    # Calculate the number of rows and percentage of rows eliminated
    rows_eliminated = initial_rows - final_rows
    percent_eliminated = (rows_eliminated / initial_rows) * 100

    # Display the filtered DataFrame and the elimination stats
    print(f"Number of rows eliminated: {rows_eliminated}")
    print(f"Percentage of rows eliminated: {percent_eliminated:.2f}%")
    
    # Unique key columns
    unique_key = ['street', 'city', 'state', 'zip_code', 'price', 'brokered_by']
    initial_size = len(df)
    # Sort DataFrame by 'prev_sold_date' in descending order
    df = df.copy().sort_values(by='prev_sold_date', ascending=False)

    # Drop duplicates based on unique key and keep the last occurrence
    df = df.copy().drop_duplicates(subset=unique_key, keep='last')
    size_no_duplicates = len(df)
    # Count the number of duplicates
    num_duplicates = size_no_duplicates - initial_size

    # Calculate the percentage of duplicates
    percent_duplicates = (num_duplicates / initial_size) * 100

    # Display the number and percentage of duplicates
    print(f"Number of duplicates: {num_duplicates}")
    print(f"Percentage of duplicates: {percent_duplicates:.2f}%")
    
    # Check table existence and insert data
    with engine.connect() as conn:
        table_exists = engine.dialect.has_table(conn, 'clean_data')
        if not table_exists:
            print("La tabla 'clean_data' no existe.")
            df.iloc[:0].to_sql('clean_data', con=engine, if_exists='replace', index=False)
        else:
            conn = mysql.connector.connect(host="mysql",user="airflow",password="airflow",database="airflow")        
            existing_batches_query = "SELECT DISTINCT batch_number FROM clean_data;"
            existing_batches = pd.read_sql_query(existing_batches_query, conn)
            existing_batches_set = set(existing_batches['batch_number'])                
            
            # Filter DataFrame to only include rows with batch_numbers not in clean_data
            df_to_insert = df[~df['batch_number'].isin(existing_batches_set)]

            # Insert data into the table
            if not df_to_insert.empty:
                df_to_insert.to_sql('clean_data', con=engine, if_exists='append', index=False, chunksize=10000)
                print("Datos insertados en 'clean_data'.")
            else:
                print("No hay nuevos datos para insertar en 'clean_data'.")
            
    # Confirm and close
    conn.commit()
    conn.close()
    
    
def perform_distribution_test(df, column, batch_num1, batch_num2):
    sample1 = df[df["batch_number"] == batch_num1][column]
    sample2 = df[df["batch_number"] == batch_num2][column]
    _, p_value = ks_2samp(sample1, sample2)
    return p_value

def train_models():

    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'supersecret'
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("mlflow_tracking_examples")

    # Conexion a la bd
    conn = mysql.connector.connect(
    host="mysql",
    user="airflow",
    password="airflow",
    database="airflow"
    )

    query = """
            WITH all_data AS (
                SELECT *,
                MAX(batch_number) OVER () AS last_batch_number
                FROM clean_data
            )
            , last_two_batch AS (
                SELECT last_batch_number, (last_batch_number - 1) AS previous_batch_number FROM all_data
            )
            SELECT
                *
            FROM all_data
            WHERE batch_number IN (SELECT last_batch_number FROM last_two_batch)
            OR batch_number IN (SELECT previous_batch_number FROM last_two_batch);
            """
    df = pd.read_sql(query, con=conn)
    conn.close()
    
    MAX_BATCH_NUMBER = max(df["batch_number"])
    PREVIOUS_MAX_BATCH_NUMBER = MAX_BATCH_NUMBER - 1

    CATEGORICAL_FEATURES = ["brokered_by",
                            "status",
                            "city",
                            "state",
                            "zip_code"]

    NUMERICAL_FEATURES = ["bed",
                        "bath",
                        "acre_lot",
                        "house_size"]

    ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

    TARGET = "price"
    
    # Calculate sizes of the batches
    size_current = len(df[df["batch_number"] == MAX_BATCH_NUMBER])
    size_previous = len(df[df["batch_number"] == PREVIOUS_MAX_BATCH_NUMBER])

    # Initialize a flag to check if any p_value is less than 0.05
    significant_difference = False

    # Condition 1
    if MAX_BATCH_NUMBER == 1:
        # Continue with the rest of the notebook
        pass
    # Condition 2
    elif size_current >= 0.1 * size_previous:
        # Perform distribution difference test for NUMERICAL_NUMBERS
        for column in NUMERICAL_FEATURES:
            p_value = perform_distribution_test(df, column, MAX_BATCH_NUMBER, PREVIOUS_MAX_BATCH_NUMBER)
            if p_value < 0.05:  # Assuming significance level of 0.05
                # At least one column has a significant difference in distribution
                # Print the column name
                print(f"Column '{column}' has a significant difference in distribution.")
                # Set the flag to True
                significant_difference = True
                # No further testing needed, break out of the loop
                break
            if not significant_difference:
                raise SystemExit()
    else:
        raise SystemExit()
    
    all_df = df[df["batch_number"] == MAX_BATCH_NUMBER]

    # Set the target values
    y = all_df['price']#.values

    # Set the input values
    X = all_df[ALL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Define preprocessing steps for categorical variables
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute with mode
        ('target_encoder', TargetEncoder())  # Target encoding
    ])

    # Define preprocessing steps for numerical variables
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Impute with median
        ('scaler', StandardScaler())  # StandardScaler
    ])

    # Combine preprocessing steps for both categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('num', numerical_transformer, NUMERICAL_FEATURES)
        ])

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    # Fit and transform the data
    X_train_preprocessed = pipeline.fit_transform(X_train, y_train)
    X_test_preprocessed = pipeline.transform(X_test)
    
    # Benchmark model

    # Train ElasticNet model with default parameters
    elasticnet_model = ElasticNet()
    elasticnet_model.fit(X_train_preprocessed, y_train)
    elasticnet_y_pred = elasticnet_model.predict(X_test_preprocessed)
    elasticnet_mae = mean_absolute_error(y_test, elasticnet_y_pred)

    # Train DecisionTreeRegressor model with default parameters
    decisiontree_model = DecisionTreeRegressor()
    decisiontree_model.fit(X_train_preprocessed, y_train)
    decisiontree_y_pred = decisiontree_model.predict(X_test_preprocessed)
    decisiontree_mae = mean_absolute_error(y_test, decisiontree_y_pred)

    # Train RandomForestRegressor model with default parameters
    randomforest_model = RandomForestRegressor()
    randomforest_model.fit(X_train_preprocessed, y_train)
    randomforest_y_pred = randomforest_model.predict(X_test_preprocessed)
    randomforest_mae = mean_absolute_error(y_test, randomforest_y_pred)

    # Choose the model with the lowest MAE
    best_model = None
    if elasticnet_mae <= decisiontree_mae and elasticnet_mae <= randomforest_mae:
        best_model = elasticnet_model
    elif decisiontree_mae <= elasticnet_mae and decisiontree_mae <= randomforest_mae:
        best_model = decisiontree_model
    else:
        best_model = randomforest_model

    print("Best model:", best_model)
    
    # Retrain the best model on the entire training data
    best_model.fit(X_train_preprocessed, y_train)
    y_pred = best_model.predict(X_test_preprocessed)
    best_model_mae = mean_absolute_error(y_test, y_pred)
    
    random_indices = np.random.choice(len(X_train_preprocessed), size=100, replace=False)
    X_subset = X_train_preprocessed[random_indices]

    # Compute SHAP values
    explainer = shap.Explainer(decisiontree_model)
    shap_values = explainer.shap_values(X_subset)
    
    # Prod model
    # Define preprocessing steps for categorical variables
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute with mode
        ('target_encoder', TargetEncoder())  # Target encoding
    ])

    # Define preprocessing steps for numerical variables
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Impute with median
        ('scaler', StandardScaler())  # StandardScaler
    ])

    # Combine preprocessing steps for both categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('num', numerical_transformer, NUMERICAL_FEATURES)
        ])

    # Create the pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_model)  # assuming best_model is already trained
    ])
    # Train
    pipeline.fit(X_train, y_train)

    # for model_name, model in models.items():
    #     pipe = Pipeline(steps=[
    #         ('preprocessor', preprocessor),
    #         ('classifier', model)
    #     ])
        
    #     param_grid = {'classifier__' + key: value for key, value in param_grids[model_name].items()}
        
    #     search = GridSearchCV(pipe, param_grid, n_jobs=-3)

    #     with mlflow.start_run(run_name=f"autolog_pipe_{model_name}") as run:
    #         search.fit(X_train, y_train)
    #         mlflow.log_params(param_grid)
    #         mlflow.log_metric("best_cv_score", search.best_score_)
    #         mlflow.log_params("best_params", search.best_params_)
    
    


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024,5,27),
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
# write_to_clean_data_task >> train_models_task
