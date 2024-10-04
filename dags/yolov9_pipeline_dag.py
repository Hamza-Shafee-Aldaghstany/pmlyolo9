from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import subprocess
def build_docker_images():
    # Build the Docker images for the API and Streamlit app
    subprocess.run(
        ["sudo","docker-compose","-f", "docker-compose.yml", "build", "--no-cache"],
        cwd="../",
        check=True
    )

def run_docker_deployment():
    # Run the composite Docker image for the API and Streamlit app
    subprocess.run(["sudo","docker-compose" ,"up"],cwd="../")
# Define your functions for each stage
def run_data_engineering():
    subprocess.run(["python", "data_processing.py"], cwd="../data")

def run_model_engineering():
    subprocess.run([
    "python", "../models/train_model.py", 
    "--workers", "2", 
    "--device", "0", 
    "--batch", "6", 
    "--data", "../data/coco.yaml", 
    "--img", "640", 
    "--cfg", "../models/detect/yolov9-t.yaml", 
    "--weights", "../weights/yolov9-t-converted.pt", 
    "--name", "yolov9-t-dep", 
    "--hyp", "hyp.scratch-high.yaml", 
    "--min-items", "0", 
    "--epochs", "2", 
    "--close-mosaic", "15"
])
# def run_api_deployment():
#     subprocess.run(["uvicorn", "../deployment/api/app:app", "--host", "0.0.0.0", "--port", "8000"])

# def run_streamlit_app():
#     subprocess.run(["streamlit", "run", "code/deployment/app/app.py"])

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 10, 1),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'yolov9_pipline',
    # dag_id='this-dag',
    default_args=default_args,
    description='An automated ML pipeline with data engineering, model training, and deployment',
    schedule_interval=None,  # Set to None for manual triggering
)

# Define the tasks in the DAG
data_engineering_task = PythonOperator(
    task_id='data_engineering',
    python_callable=run_data_engineering,
    dag=dag,
)

model_engineering_task = PythonOperator(
    task_id='model_engineering',
    python_callable=run_model_engineering,
    dag=dag,
)
build_images_task = PythonOperator(
    task_id='build_docker_images',
    python_callable=build_docker_images,
    dag=dag,
)

# Create a task for deploying the API and Streamlit app
docker_deployment_task = PythonOperator(
    task_id='docker_deployment',
    python_callable=run_docker_deployment,
    dag=dag,
)
# api_deployment_task = PythonOperator(
#     task_id='api_deployment',
#     python_callable=run_api_deployment,
#     dag=dag,
# )

# streamlit_app_task = PythonOperator(
#     task_id='streamlit_app',
#     python_callable=run_streamlit_app,
#     dag=dag,
# )

# Set the task dependencies
data_engineering_task >> model_engineering_task >> build_images_task >> docker_deployment_task
if __name__ == "__main__":
    dag.test()