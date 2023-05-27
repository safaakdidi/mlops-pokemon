import mlflow

def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri("http://localhost:5000")  
    mlflow.set_experiment(experiment_name)
    
