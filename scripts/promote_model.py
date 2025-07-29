import mlflow
import dagshub
from mlflow.tracking import MlflowClient

def promote_model():
    # Initialize DagsHub MLflow tracking
    dagshub.init(
        repo_owner='yahiaehab10', 
        repo_name='MLFlow_demo', 
        mlflow=True
    )
    
    client = MlflowClient()
    
    # Get the latest model version
    model_name = "IrisRandomForest"
    latest_versions = client.get_latest_versions(
        model_name, 
        stages=["Staging"]
    )
    
    if latest_versions:
        latest_version = latest_versions[0]
        
        # Check if model meets production criteria
        run = client.get_run(latest_version.run_id)
        accuracy = run.data.metrics.get('accuracy', 0)
        
        if accuracy > 0.9:  # Production threshold
            # Promote to production
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Production"
            )
            print(f"Model version {latest_version.version} promoted to Production")
        else:
            print(f"Model accuracy {accuracy} below production threshold")

if __name__ == "__main__":
    promote_model()
