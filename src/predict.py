import pandas as pd


def predict_default(model,customer_data,threshold=0.45):
    X=pd.DataFrame([customer_data])

    expected_features=list(model.feature_names_in_)
    if set(X.columns)!=set(expected_features):
        raise ValueError("Customer input fields do not match.")

    X=X[expected_features]
    probability=model.predict_proba(X)[0,1]
    flag=probability>=threshold

    return {
        "default_probability": float(probability),
        "flag_for_review":bool(flag)
    }

