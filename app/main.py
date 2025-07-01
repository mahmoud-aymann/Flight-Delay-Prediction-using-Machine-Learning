import streamlit as st
import pandas as pd
import joblib

st.title("Flight Delay Prediction App")
model = joblib.load(open(r'models\flight_delay_rf_tuned.pkl', 'rb'))

year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
carrier_code = st.text_input("Carrier Code (e.g., AA, DL)")
airport_code = st.text_input("Airport Code (e.g., JFK, LAX)")
arrival_flights = st.number_input("Arrival Flights", min_value=0)
weather_delay_count = st.number_input("Weather Delay Count", min_value=0)
security_delay_count = st.number_input("Security Delay Count", min_value=0)
arrival_cancelled_count = st.number_input("Arrival Cancelled Count", min_value=0)
arrival_diverted_count = st.number_input("Arrival Diverted Count", min_value=0)
delay_rate = st.number_input("Delay Rate (e.g., 0.1 for 10%)", min_value=0.0, max_value=1.0, step=0.01)

input_data = pd.DataFrame({
    'year': [year],
    'carrier_code': carrier_code,
    'airport_code':airport_code,
    'arrival_flights': [arrival_flights],
    'weather_delay_count': [weather_delay_count],
    'security_delay_count': [security_delay_count],
    'arrival_cancelled_count': [arrival_cancelled_count],
    'arrival_diverted_count': [arrival_diverted_count],
    'delay_rate': [delay_rate]
})
if st.button("Predict Delay"):
    prediction = model.predict(input_data)
    st.write(f"Predicted delay: {prediction[0]}")