# Credit Default Risk Prediction System

## Project Overview

This is an end-to-end machine learning project built to estimate the probability of credit_card defaults using historical data of customer and their repayment information.

The system here uses a Logistic Regression pipeline with the preprocessing of features, a FastApi inference service, automated tests and a Dockers containerization.

## Dataset

The dataset used here is the UCI Default of Credit Card Clients with about 30000 records from Taiwan in 2005.

 The dataset contains 23 features which includes demographic, credit-limit, repayment-status, billing and payment information of the customer. The target feature is **default payment next month**.

For the development of the model the data was split into stratified sets of training (64%), validation (16%), and testing (20%). After the model selection the training and the validation set were combined while the remaining 20% was kept for held-out testing.

Source: [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## Data Preprocesing

The 23 features were divided into three groups for preprocessing.
**Numerical Features(14): Credit limits,age,billing amounts and payment amounts were standardized using StandardScaler.
**Demographic Features(3): Sex, Education and marital status were encoded using OneHotEncoder. 
**Repayment-Status Features(6)**: To avoid the repayment features to be understood as a continuous measurements all six months of history was OneHotEncoded.

These features were preprocessed using the Scikit-learn's ColumnTransformer. The preprocessing steps and the model were combined into a single Pipeline to ensure consistent transformations during training as well as during inference while preventing data leakage during model evaluation.

### Model Development and Selection

Model development began with a DummyClassifier to establish a baseline, followed by Logistic Regression as the first predictive model due to its simplicity and interpretability.

Random Forest and XGBoost were subsequently evaluated to determine whether more complex models could capture nonlinear relationships and improve predictive performance.

The models were compared using stratified cross-validation and a separate validation set, with particular emphasis on Average Precision and ROC-AUC. An unrestricted Random Forest exhibited substantial overfitting, which was reduced by limiting tree depth.

Although Random Forest and XGBoost achieved slightly higher ranking metrics, their improvements over Logistic Regression were modest. Logistic Regression was therefore selected as the final model because it offered comparable predictive performance with lower complexity and more straightforward interpretation.

## Threshold Selection

### Threshold Selection

Multiple classification thresholds were evaluated to understand the trade-off between precision, recall, and the number of customers flagged for review.

Lowering the threshold increased recall by identifying more actual defaulters, but it also increased false positives and reduced precision. A threshold of **0.45** was selected as an illustrative operating point for a hypothetical manual credit-risk review process, balancing the ability to identify defaults against the number of unnecessary reviews.

The model's predicted probabilities were separately assessed using the Brier score and calibration curve. These evaluations helped assess probability quality but were not used to directly optimize the classification threshold.

The selected threshold is not a business-validated optimum and would require further analysis of review capacity, error costs, and operational requirements before real-world use.

## Model Performance

Accuracy:	        81.73%
Precision:	        64.90%
Recall:	            37.91%
F1 Score:	        0.4786
ROC-AUC:	        0.7583
Average Precision:	0.5277
Brier Score:	    0.1387
Threshold:	        0.45

The model demonstrated consistent performance between validation and held-out test datasets, suggesting reasonable generalization within this historical dataset. However, recall remained relatively low, meaning a substantial proportion of actual defaulters were not flagged.

## System Architecture

The system is divided into two workflows: **model training** and **model inference**.

### Training Workflow

`train.py` coordinates the training process by calling `data.py` to load and split the dataset and `model.py` to construct the preprocessing and Logistic Regression pipeline. The pipeline is fitted on the training data and then serialized using `joblib`, allowing it to be reused without retraining.

### Inference Workflow

`app.py` initializes the FastAPI application and loads the saved model. Incoming customer information is received as JSON and validated using a Pydantic schema. The validated input is converted into a dictionary using `.model_dump()` and passed to `predict.py`, which prepares the features, calls the fitted pipeline, applies the 0.45 threshold, and returns the predicted default probability and review flag as JSON.

The inference application is also containerized using Docker to provide a consistent runtime environment.


## Installation and Usage

### Prerequisites

- Python 3.14 (the project was developed using Python 3.14.6)
- Docker Desktop (optional, for containerized execution)
- Git

### 1. Set Up the Python Environment

From the project root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### 2. Prepare the Dataset

Download the [Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) from the UCI Machine Learning Repository.

Place the original Excel file at:

```text
data/raw/default of credit card clients.xls
```

### 3. Train the Model

Run:

```bash
python train.py
```

This loads the historical dataset, creates a stratified 80/20 train-test split, trains the preprocessing and Logistic Regression pipeline on the training portion, and saves the fitted model using `joblib`.

### 4. Start the FastAPI Application

```bash
python -m uvicorn app:app --reload
```

Visit:

- API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

Use the `POST /predict` endpoint to submit a customer record containing the 23 required input features.

The API returns a predicted default probability and a Boolean indicating whether the customer is flagged for manual review using a threshold of 0.45.

### 5. Run Using Docker

Alternatively, build the Docker image from the project root:

```bash
docker build -t credit-default-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 credit-default-api
```

The application will be accessible at http://127.0.0.1:8000/docs.

The container uses the saved model artifact and does not retrain the model on startup.

