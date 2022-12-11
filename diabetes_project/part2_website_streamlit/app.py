import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

st.header("""Diabetes Prediction App""")

def user_input_features():
        Pregnancies = st.sidebar.slider('Pregnancies', 0, 1, 10)
        Glucose = st.sidebar.slider('Glucose', 0, 120, 250)
        BloodPressure = st.sidebar.slider('BloodPressure', 0, 80, 150)
        SkinThickness = st.sidebar.slider('SkinThickness', 0, 20, 100)
        Insulin = st.sidebar.slider('Insulin', 0, 30, 1000)
        BMI = st.sidebar.slider('BMI', 0, 30, 80)
        DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0, 2, 5)
        Age = st.sidebar.slider('Age', 0, 30, 100)

        data = {'Pregnancies': Pregnancies,
                'Glucose': Glucose,
                'BloodPressure': BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
                'Age': Age}
        features = pd.DataFrame(data, index=[0])
        return features

df = user_input_features()

# Input
st.subheader('User Input')

st.write(df)

# Call classification model
load_clf = pickle.load(open('diabetes_final_RF_model.pkl', 'rb'))

# Make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

if prediction[0] == 0:
        st.subheader('Prediction:')
        st.write('Non Diabetes')
        st.subheader('Prediction Probability')
        st.write(prediction_proba[0][0])
else:
        st.subheader('Prediction:')
        st.write('Diabetes')
        st.subheader('Prediction Probability')
        st.write(prediction_proba[0][1])