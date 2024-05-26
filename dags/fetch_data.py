from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import requests
import pandas as pd
from sqlalchemy import create_engine
import logging

COLUMN_NAMES = [
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

# Configuración del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data():
    url = "http://10.43.101.149/data"
    params = {'group_number': '2'}
    headers = {'accept': 'application/json'}

    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        json_data = response.json()
        df = pd.DataFrame.from_dict(json_data["data"])
        df.columns = COLUMN_NAMES
        
        # Connect to MySQL and create table if not exists
        engine = create_engine("mysql+mysqlconnector://airflow:airflow@mysql/airflow")
        with engine.connect() as conn:
            table_exists = engine.dialect.has_table(conn, 'data_table') # Replace 'your_table_name' with actual table name
            if not table_exists:
                print("No existe la tabla")
                df.iloc[:0].to_sql('data_table', con=engine, if_exists='replace', index=False) # Create empty table with correct structure

            # Merge data into the table
            df.to_sql('data_table', con=engine, if_exists='append', index=False)
            
    else:
        logger.error("Error al realizar la solicitud: %d", response.status_code)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023,3,4),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

schedule_interval = "*/5 * * * *"

dag = DAG(
    'Data_fetch_dag',
    default_args=default_args,
    description='Fetch data every 5 minutes',
    schedule_interval=schedule_interval,
    catchup=False
)

fetch_data_task = PythonOperator(
    task_id='fetch_data_task',
    python_callable=fetch_data,
    dag=dag,
)

fetch_data_task
