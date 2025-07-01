import os 
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from preprocess import prepare_data
"""
This module handles model training for flight delay prediction.
"""
def split_data(X,y,test_size=0.2,random_state=42):
    """
    Split data into training and testing sets.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility
    Returns:
    tuple:X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
def train_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train and evaluate multiple models.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        preprocessor: Data preprocessing pipeline
        
    Returns:
        dict: Results dictionary with model metrics
    """
    models={
    "LinerRegression":LinearRegression(),
    "DecisionTree":DecisionTreeRegressor(),
    "RandomForest":RandomForestRegressor(random_state=42),
    }
    results={}
    for model_name, model in models.items():
        pipeline=Pipeline(steps=[
            ("precocessor",preprocessor),
            ("regressor",model)
        ])
        pipeline.fit(X_train,y_train)
        y_test_pred=pipeline.predict(X_test)
        y_train_pred=pipeline.predict(X_train)
        train_mae=mean_absolute_error(y_train,y_train_pred)
        train_r2=r2_score(y_train,y_train_pred)
        test_mae=mean_absolute_error(y_test,y_test_pred)
        test_r2=r2_score(y_test,y_test_pred)
        results[model_name]={
            "Train MAE":train_mae,
            "Test MAE":test_mae,
            "Train R²":train_r2,
            "Test R²":test_r2
        }
        print(f"\n{model_name}")
        print("Training Mean Absolute Error (MAE):", train_mae)
        print("Testing Mean Absolute Error (MAE):", test_mae)
        print("Train R²:", train_r2)
        print("Test R²:", test_r2)
    return results
def tune_random_forest(X_train, y_train, preprocessor):
    """
    Perform hyperparameter tuning for RandomForest.
    
    Args:
        X_train, y_train: Training data
        preprocessor: Data preprocessing pipeline
        
    Returns:
        Pipeline: Best model pipeline
    """
    # Create base pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])
    
    # Define parameter grid
    param_grid = {
        "regressor__n_estimators": [100, 300],
        "regressor__max_depth": [30],
    }
    
    # Perform randomized search
    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    print("Performing hyperparameter tuning for RandomForest...")
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters:", grid_search.best_params_)
    
    return grid_search.best_estimator_
def save_model(model,file_path):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        file_path (str): Path to save the model
    """
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    joblib.dump(model,file_path)
    print(f"Model saved to {file_path}")
def main():
    """
    Main function to run the training pipeline.
    """
    data_path=os.path.join("data","airline_delay.csv")
    model_dir=os.path.join("models")
    os.makedirs(model_dir,exist_ok=True)
    # prepare data
    print("Preparing data...")
    X, y, preprocessor = prepare_data(data_path)
    # split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Train and evaluate multiple models
    print(X_train.columns,X_test.columns)
    print("Training and evaluating models...")
    results = train_models(X_train, y_train, X_test, y_test, preprocessor)
    best_model_name=max(results,key=lambda k:results[k]["Test R²"])
    print(f"\n Best model:{best_model_name} with Test R²: {results[best_model_name]['Test R²']}")
    if best_model_name=="RandomForest":
        print("\nTuning RandomForest hyperparameters...")
        tuned_model = tune_random_forest(X_train, y_train, preprocessor)
        y_test_pred = tuned_model.predict(X_test)
        y_train_pred = tuned_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print("\nTuned RandomForest")
        print("Training Mean Absolute Error (MAE):", train_mae)
        print("Testing Mean Absolute Error (MAE):", test_mae)
        print("Train R²:", train_r2)
        print("Test R²:", test_r2)
        print(os.path.join(model_dir,f'flight_delay_rf_tuned.pkl'))
        save_model(tuned_model,os.path.join(model_dir,f'flight_delay_rf_tuned.pkl'))
if __name__ == "__main__":
    main()
