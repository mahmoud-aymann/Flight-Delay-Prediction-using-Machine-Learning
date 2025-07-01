# ✈️ Flight Delay Prediction

This project analyzes and predicts flight delays based on various features like carrier information, airport details, and flight statistics.

---

## 📁 Project Structure

| Directory/File               | Description                          |
|------------------------------|--------------------------------------|
| `ml_project/data/`           | Contains dataset files               |
| └── `airline_delay.csv`      | Flight delay dataset                 |
| `ml_project/notebooks/`      | Exploratory analysis notebooks       |
| └── `Flight_Delay_Dataset_Description.ipynb` | EDA notebook         |
| `ml_project/src/`            | Source code                          |
| ├── `__init__.py`            | Package initialization               |
| ├── `preprocess.py`          | Preprocessing pipeline               |
| ├── `train.py`               | Training script                      |
| └── `predict.py`             | Prediction logic                     |
| `ml_project/requirements.txt`| Project dependencies                 |
| `ml_project/README.md`       | Project documentation                |

---

## 📊 Dataset Description

The dataset contains information about flight delays at various airports, including:

- **Date Information:** Year and month  
- **Carrier Information:** Airline carrier code and name  
- **Airport Information:** Airport code and name  
- **Flight Statistics:** Total arrivals, cancellations, diversions, and delays  
- **Delay Cause Counts:** Number of flights delayed by each cause category  
- **Delay Durations:** Minutes of delay for different causes  

---

## ⚙️ Features

- Data preprocessing pipeline  
- Exploratory data analysis  
- Flight delay prediction using machine learning  
- Model evaluation and comparison  
- Feature importance analysis  

---

## 🤖 Models

The project implements and compares multiple regression models:

- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor (**best performing**)  

---

## 📈 Performance

The optimized Random Forest model achieves:

- **R² score:** ~0.99 on test data  
- **Mean Absolute Error (MAE):** ~1.9 on test data  

---


