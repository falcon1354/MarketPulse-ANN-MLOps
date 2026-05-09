from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def ingestion():
    print("Running Data Ingestion")

def preprocessing():
    print("Running Preprocessing")

def training():
    print("Training Model")

with DAG("marketpulse_pipeline",
         start_date=datetime(2024,1,1),
         schedule_interval="@daily",
         catchup=False) as dag:

    task1 = PythonOperator(task_id="ingestion", python_callable=ingestion)
    task2 = PythonOperator(task_id="preprocessing", python_callable=preprocessing)
    task3 = PythonOperator(task_id="training", python_callable=training)

    task1 >> task2 >> task3