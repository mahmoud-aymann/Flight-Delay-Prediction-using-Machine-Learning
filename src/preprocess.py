import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
"""
This module contains data preprocessing functions for flight delay prediction.
"""
def load_data(file_path):
    """
    load the fight delay dataset and rename columns for clarity.
    Args:
        file_path (str): Path to the CVS file
    Returns:
    pd.DataFrame: loaded dataset and renamed dataframe
    """
    # Load the dataset
    df=pd.read_csv(file_path)
    # Rename columns for clarity
    df.rename(columns={
    'year': 'year',
    'month': 'month',
    'carrier': 'carrier_code',
    'carrier_name': 'carrier_name',
    'airport': 'airport_code',
    'airport_name': 'airport_name',
    'arr_flights': 'arrival_flights',
    'arr_del15': 'arrival_delay_over_15min',
    'carrier_ct': 'carrier_delay_count',
    'weather_ct': 'weather_delay_count',
    'nas_ct': 'nas_delay_count',
    'security_ct': 'security_delay_count',
    'late_aircraft_ct': 'late_aircraft_delay_count',
    'arr_cancelled': 'arrival_cancelled_count',
    'arr_diverted': 'arrival_diverted_count',
    'arr_delay': 'total_arrival_delay_minutes',
    'carrier_delay': 'carrier_delay_minutes',
    'weather_delay': 'weather_delay_minutes',
    'nas_delay': 'nas_delay_minutes',
    'security_delay': 'security_delay_minutes',
    'late_aircraft_delay': 'late_aircraft_delay_minutes'
    },inplace=True)
    return df
def clean_data(df):
    """
    Clean the dataset by handling missing values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Remove rows with missing values
    df = df.dropna()
    
    return df
def engineer_features(df):
    """
    Create new features  and drop unnecessary columns.

    Args:
        df (pd.DataFrame): Input dataframe
    Returns:
    pd.DataFrame: Dataframe with engineered features
    """
    df["delay_rate"]=df["arrival_delay_over_15min"]/df["arrival_flights"]
    # Drop unnecessary columns
    columns_to_drop=[
    'carrier_name',
    'airport_name',
    'month',
    'total_arrival_delay_minutes',
    'carrier_delay_minutes',
    'weather_delay_minutes',
    'nas_delay_minutes',
    'security_delay_minutes',
    'late_aircraft_delay_minutes']
    df=df.drop(columns=columns_to_drop)
    return df

def handle_multicollinearity(df, threshold=0.8):
    """
    Identify and remove highly correlated features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Correlation threshold for dropping columns
        
    Returns:
        pd.DataFrame: Dataframe with reduced multicollinearity
    """
    # Calculate correlation matrix for numerical columns
    corr_matrix = df.corr(numeric_only=True)
    
    # Create mask for upper triangle
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_corr = pd.DataFrame(corr_matrix.values, columns=corr_matrix.columns,index=corr_matrix.index)
    upper_corr = upper_corr.where(upper)
    
    # Find columns to drop based on correlation threshold
    to_drop = [column for column in upper_corr.columns if any(upper_corr[column] > threshold)]
    
    return df.drop(columns=to_drop)
def create_preprocessor(df):
    """
    Create a scikit-learn preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe to determine column types
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Remove target column if it's in numerical_columns
    if "arrival_delay_over_15min" in numerical_columns:
        numerical_columns.remove("arrival_delay_over_15min")
    
    # Create preprocessing pipeline
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_columns),
            ("num", numerical_transformer, numerical_columns)
        ]
    )
    
    return preprocessor
def prepare_data(file_path):
    """
    Complete data preparation pipeline.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: X (features), y (target), preprocessor
    """
    df=load_data(file_path)
    df=clean_data(df)
    df=engineer_features(df)
    print(df.columns)
    X=df.drop(columns="arrival_delay_over_15min")
    y=df["arrival_delay_over_15min"]
    X=handle_multicollinearity(X)
    preprocessor=create_preprocessor(X)
    print()
    return X,y,preprocessor

