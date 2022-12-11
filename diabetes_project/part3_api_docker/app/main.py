from fastapi import FastAPI
from fastapi import Request, Form
import numpy as np
import pickle
import joblib
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

@app.get('/')
def main():
    return {'Diabetes Prediction API'}

@app.get('/predict')
def predict(
    Pregnancies: float,
    Glucose: float,
    BloodPressure: float,
    SkinThickness: float,
    Insulin: float,
    BMI: float,
    DiabetesPedigreeFunction: float,
    Age: float):

    dic = {
    'Pregnancies': Pregnancies,
    'Glucose': Glucose,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age
    }

    df = pd.DataFrame(dic, index=[0])

    load_clf = pickle.load(open('diabetes_final_RF_model.pkl', 'rb'))

    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)

    pred_prob = prediction_proba[0][1]

    result = 'Your Diabetes Risk: ' + str(pred_prob)

    return result