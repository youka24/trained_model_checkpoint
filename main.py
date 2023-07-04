import numpy as np
import pickle
import streamlit as st
import sklearn
# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


def diabetes_prediction(inputdata):
    inputdata_as_numpy_array = np.asarray(inputdata)
    input_data_resheaped = inputdata_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_resheaped)
    print(prediction)
    return prediction


st.title('Welcome to COVID-19 Prediction')
death = st.number_input('Number of death')
deathRate = st.number_input('deathRate')
confirmedRate = st.number_input('confirmedRate')
year = st.number_input('year')
Month = st.number_input('Month')
diagnosis = ''

# creating a button for Prediction

if st.button('Diabetes Test Result'):
    diagnosis = diabetes_prediction([death, deathRate, year, Month, confirmedRate])

st.success(diagnosis)
