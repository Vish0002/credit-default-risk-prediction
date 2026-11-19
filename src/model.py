from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression

def build_pipeline():
    numeric_features = ["LIMIT_BAL","AGE",'BILL_AMT1', 'BILL_AMT2','BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    repayment_features = [ 'PAY_0', 'PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    preprocessor=ColumnTransformer(transformers=[("num",StandardScaler(),numeric_features),
                                                 ("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features),
                                                 ("repf",OneHotEncoder(handle_unknown="ignore"),repayment_features)])

    model=Pipeline([
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(max_iter=1000))
    ])


    return model