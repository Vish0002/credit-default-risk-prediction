import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    data=pd.read_excel(path,header=1)

    X=data.drop(columns=["ID","default payment next month"])
    y=data["default payment next month"]

    return X,y

def split_data(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return X_train,X_test,y_train,y_test


