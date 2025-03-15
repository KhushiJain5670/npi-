
import streamlit as st

st.title('Doctor Survey Response Predictor')

st.sidebar.header('Enter a time to predict (HH:MM)')
time_input = st.sidebar.text_input('Time (e.g., 14:00)', '14:00')

st.write('You entered:', time_input)
    