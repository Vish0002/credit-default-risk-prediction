from pathlib import Path
import joblib

from src.data import load_data,split_data
from src.model import build_pipeline


PROJECT_ROOT= Path(__file__).resolve().parent 

def main():
    data_path=PROJECT_ROOT / "data/raw/default of credit card clients.xls"
    X,y=load_data(data_path)
    X_train,X_test,y_train,y_test=split_data(X,y)

    model=build_pipeline()
    model.fit(X_train,y_train)

    model_path=PROJECT_ROOT / "model"/"credti_default_pipeline.joblib"
    model_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,model_path)
    print(f"Model Saved to:{model_path}")

if __name__=="__main__":
    main()