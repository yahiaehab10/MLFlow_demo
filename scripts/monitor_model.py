import mlflow
import dagshub
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import accuracy_score

def monitor_model_performance():
    """Monitor model performance in production"""
    dagshub.init(
        repo_owner='yahiaehab10', 
        repo_name='MLFlow_demo', 
        mlflow=True
    )
    
    client = MlflowClient()
    
    # Get production model
    model_name = "IrisRandomForest"
    production_models = client.get_latest_versions(
        model_name, 
        stages=["Production"]
    )
    
    if production_models:
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Load new production data
        new_data = pd.read_csv("data/processed/new_production_data.csv")
        X_new = new_data.drop('target', axis=1)
        y_new = new_data['target']
        
        # Make predictions
        predictions = model.predict(X_new)
        accuracy = accuracy_score(y_new, predictions)
        
        # Log performance
        with mlflow.start_run(run_name="production_monitoring"):
            mlflow.log_metric("production_accuracy", accuracy)
            
            if accuracy < 0.8:  # Performance degradation threshold
                mlflow.log_param("performance_alert", True)
                print("ALERT: Model performance degraded!")

if __name__ == "__main__":
    monitor_model_performance()
