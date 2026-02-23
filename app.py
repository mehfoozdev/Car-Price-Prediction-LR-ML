from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('car_price_lr_model.pkl','rb'))
car = pd.read_csv('car_data_cleaned.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('home.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))  
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven')) 
    
    # Create DataFrame with correct column order
    input_data = pd.DataFrame([[car_model, company, year, driven, fuel_type]], 
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    
    print("Input data:", input_data)
    
    prediction = model.predict(input_data)[0]
    print("Prediction:", prediction)
    
    # Redirect back with prediction as query parameter
    return render_template('home.html', prediction=prediction)


if __name__=='__main__':
    app.run()