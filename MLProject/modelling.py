import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

DAGSHUB_USERNAME = "laventilizz"
REPO_NAME = "Eksperimen_SML_zaza"
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=REPO_NAME, mlflow=True)
             
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'water_preprocessing/train_clean.csv')
    test_path = os.path.join(base_dir, 'water_preprocessing/test_clean.csv')

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

    # setup mlflow local & autolog
    mlflow.set_experiment("Water_Quality")

    print("start training")
    
    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Training Selesai! Akurasi: {acc}")

if __name__ == "__main__":
    train_model()