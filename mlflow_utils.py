import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def setup_experiment(experiment_name="Hand-Gesture-Classification"):
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)

def log_dataset(df, path="hand_landmarks_data.csv"):
    mlflow.log_param("dataset_rows", df.shape[0])
    mlflow.log_param("dataset_cols", df.shape[1])
    mlflow.log_param("num_classes", df['label'].nunique())
    mlflow.log_artifact(path, artifact_path="dataset")

def log_model_params(model_name, params: dict):
    mlflow.log_param("model_name", model_name)
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def log_artifact(file_path, artifact_path=None):
    mlflow.log_artifact(file_path, artifact_path=artifact_path)

def log_model(model, model_name, X_sample, y_pred_sample):
    signature = infer_signature(X_sample, y_pred_sample)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name=None 
    )

def register_best_model(run_id, model_name="Best-Hand-Gesture-Classifier"):
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered: {model_name} from run {run_id}")