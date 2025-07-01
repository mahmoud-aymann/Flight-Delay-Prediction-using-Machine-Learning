"""
This module handles making predictions using a trained Random Forest model.
"""
import os 
import joblib
import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from .preprocess import load_data, clean_data, engineer_features,handle_multicollinearity,create_preprocessor

def load_model(model_path):
    """
    Loads a trained model from disk.
    """
    return joblib.load(model_path)
def prepare_new_data(file_path):
    """
    Load and preprocess the new dataset.
    """
    df=load_data(file_path)
    df=clean_data(df)
    df=engineer_features(df)
    print("Prepared features:",df.columns)
    X=df.drop(columns="arrival_delay_over_15min")
    y=df["arrival_delay_over_15min"]
    X=handle_multicollinearity(X)
    preprocessor=create_preprocessor(X)
    
    return X,y
    
def  make_predictions(model, X):
    """
    Make predictions using the trained model.
    """
    return model.predict(X)

def evaluate_predictions(y_true,y_pred):
    """
    Evaluate the predictions using metrics.
    """
    from sklearn.metrics import mean_absolute_error,r2_score
    return {
        "MAE":mean_absolute_error(y_true,y_pred),
        "R²":r2_score(y_true,y_pred)
    }
def main():
    data_path=os.path.join("data","airline_delay.csv")
    model_path=os.path.join("models","flight_delay_rf_tuned.pkl")
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return 
    print(f"Loading model from {model_path}....")
    model=load_model(model_path)
    print("Model loaded successfully.")
    print("Preparing data ....")
    X,y=prepare_new_data(data_path)
    print("Data prepared successfully.")
    print("Making predictions ....")
    predictions=make_predictions(model,X)
    print("Predictions made successfully.")
    if y is not None:
        print("Evaluating predictions ....")
        metrics=evaluate_predictions(y,predictions)
        print("Evaluating predictions\n",metrics)
    output_dir=os.path.join("results")
    os.makedirs(output_dir,exist_ok=True)    
    
    output_df= pd.DataFrame({
        "predictions":predictions,
        "actual":y.values,
        "error": y.values-predictions
    })
    output_file=os.path.join(output_dir,"predictions.csv")
    output_df.to_csv(output_file,index=False)
    print(f"Predictions saved to {output_file}")
if __name__ == "__main__":
    main()
