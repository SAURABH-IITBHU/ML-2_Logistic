from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np



application = Flask(__name__)
app=application


regressor=pickle.load(open('Model/prediction.pkl', 'rb'))
standard_scaler=pickle.load(open('Model/StandardScaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    if request.method=='POST':
        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))


        scaled_data=standard_scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        result=regressor.predict(scaled_data)

        if result[0] == 0:
            test = 'Non Diabetic'
        else:
            test = 'Diabetic'


        return render_template('prediction.html', results=test)


    else:
        return render_template('home.html') 




if __name__=="__main__":
    app.run(host="0.0.0.0")
