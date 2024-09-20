from flask import Flask, render_template, request
import requests
ALLOWED_EXTENSIONS=set(['csv'])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
app = Flask(__name__)
@app.route('/')
def main():
    return render_template('home.html')
@app.route('/predict')
def predict():
    return render_template('index.html')
@ app.route('/crop-pred')
def crop_pred():
    return render_template('index.html')
@ app.route('/crop-yield', methods=['POST'])
def crop_yield():
  if request.method == 'POST':
    crop_input = request.form.get("crop")
    area = int(request.form['area'])
    season = int(request.form['season'])
    state = request.form.get("stt")
    city = request.form.get("city")
    Dis = city.strip().upper()
    State = state.upper()

    data = pd.read_csv("C:\\Users\\alamp\\Documents\\A20_project\\A20_project\\dataset.csv")
    data.head(7)
    df = data.copy()
    
    df.dropna(axis=0, inplace=True)
    df["Crop_Name"].value_counts()
    crop_count = df["Crop_Name"].value_counts()
    df = df.loc[df["Crop_Name"].isin(crop_count.index[crop_count > 1500])]
    crop_name = crop_input.title()
    crop = df[(df["Crop_Name"] == crop_name)]
    crop.head()
    dt = crop.copy()
    le = LabelEncoder()
    dt['State_Name'] = dt['State_Name'].str.upper()
    dt["district"] = le.fit_transform(dt["City_Name"])
    dt['season'] = le.fit_transform(dt["Season"])
    dt["state"] = le.fit_transform(dt["State_Name"])
    X = dt[["Area", "district", "season", "state"]]
    y = dt["Production"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.15)
    best=None
    best_mse=float('inf')
    models = [
    LinearRegression(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    ]
    for model in models:
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)
      mse = mean_absolute_error(y_test, y_pred)
     # print(f"Mean absolute Error for {model.__class__.__name__}: {mse}")
    # Plot a bar graph comparing the MSE values of each model
      if mse < best_mse:
        best = model
        best_mse = mse
    my_dict = pd.Series(dt.City_Name.values, index=dt.district).to_dict()
    key_list = list(my_dict.keys())
    val_list = list(my_dict.values())
    position = val_list.index(Dis)
    district_id = key_list[position]
    state_id = dt[dt.State_Name == State]['state'].values[0]
    x = [[area, district_id, season, state_id]]
    ynew = best.predict(x)
    prediction = (ynew[0])  
    yield_by_area = prediction / area
    return render_template('result.html', yield_by_area=yield_by_area)
  else:
    return render_template('try_again.html')
if __name__ == "__main__":
    app.run(debug=True)
@app.route('/save', methods=['POST'])
def save():
    data = request.form['character']
    response = render_template('index.html')
    return response    
