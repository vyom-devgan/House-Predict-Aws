# California Housing Price Prediction App

## Overview
This Streamlit application predicts median house prices in California using a Random Forest model trained on the California Housing dataset. 

## Features
- Predict house prices based on features such as income, age, population, and location.
- Intuitive and interactive user interface for easy input of feature values.
- Real-time predictions and model performance metrics.

## Installation
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd california-housing-app
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the app:
    ```bash
    streamlit run app.py
    ```

## Deployment
### Docker
1. Build the Docker image:
    ```bash
    docker build -t california-housing-app .
    ```
2. Run the container:
    ```bash
    docker run -p 8501:8501 california-housing-app
    ```

### AWS ECS
Follow the deployment guide provided in the documentation to deploy the app on AWS ECS and access it via the service endpoint.

## Model Details
- Dataset: California Housing Dataset (scikit-learn)
- Algorithm: Random Forest Regressor
- Metrics: MAE, MSE, R² Score