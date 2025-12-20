import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import time

DAGSHUB_USERNAME = "laventilizz"
REPO_NAME = "Eksperimen_SML_zaza"

remote_server_uri = os.getenv('MLFLOW_TRACKING_URI', default=f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow")
mlflow.set_tracking_uri(remote_server_uri)
print(mlflow.get_tracking_uri())
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
             
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'water_preprocessing/train_clean.csv')
    test_path = os.path.join(base_dir, 'water_preprocessing/test_clean.csv')
    run_id_path = os.path.join(base_dir, 'run_id.txt')
    local_model_path = os.path.join(base_dir, 'model')

    if not os.path.exists(train_path):
        print(f"Error: File {train_path} not found")
        return

    print("loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = 'Potability'
    
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    print("start training")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Active Run ID: {run_id}")

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        clf.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Akurasi: {acc}")
        mlflow.log_metric("accuracy", acc)

        print("Uploading to DagsHub...")
        mlflow.sklearn.log_model(clf, "model")
        
        print(f"Saving locally to {local_model_path}...")
        mlflow.sklearn.save_model(clf, local_model_path)
        print("Model successfully logged")

        print(f"Saving Run ID to: {run_id_path}")
        with open(run_id_path, "w") as f:
            f.write(run_id)

if __name__ == "__main__":
    train_model()