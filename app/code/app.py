# Importing libraries
from flask import Flask, render_template, request, flash
import pickle
import numpy as np
from linear_regression import LinearRegression, NormalPenalty, LassoPenalty, RidgePenalty, ElasticPenalty, Normal, Lasso, Ridge, ElasticNet


# Setting app to run on Flask
app = Flask(__name__)

# Setting a secret key to use flash
app.secret_key = 'ml2023'


# Importing the old model
filename1 = 'model/car_price_old.model'
loaded_data1 = pickle.load(open(filename1, 'rb'))

# Separating the values in the old model file into variables for easy access
model_old = loaded_data1['model']
scaler_old = loaded_data1['scaler']
name_map_old = loaded_data1['name_map']
engine_default_old = loaded_data1['engine_default']
mileage_default_old = loaded_data1['mileage_default']

# Importing the new model
filename2 = 'model/car_price_new.model'
loaded_data2 = pickle.load(open(filename2, 'rb'))

# Separating the values in the new model file into variables for easy access
model_new = loaded_data2['model']
scaler_new = loaded_data2['scaler']
name_map_new = loaded_data2['name_map']
engine_default_new = loaded_data2['engine_default']
mileage_default_new = loaded_data2['mileage_default']

# The home page containing a link to the prediction page
@app.route('/')
def index():
    return render_template('index.html')

# The prediction page
@app.route('/predict_old')
def predict_old():
    return render_template('predict_old.html')

# The route to calculate the prediction result but not accessed by users
@app.route('/process-data_old', methods = ['POST'])
def process_data_old():
    if request.method == 'POST':
        # Getting the values required for prediction
        brand_name = request.form.get('name')
        name = name_map_old.get(brand_name,'32')
        engine = request.form.get('engine', engine_default_old)
        mileage = request.form.get('mileage', mileage_default_old)

        # Convert engine and mileage to float only if they are not empty strings
        if engine:
            engine = float(engine)
        else:
            engine = engine_default_old  # Set a default value if engine is empty

        if mileage:
            mileage = float(mileage)
        else:
            mileage = mileage_default_old # Set a default value if mileage is empty

        # Calling the prediction function, coverting the result to int for user experience and then to string
        # to display on the website
        result = str(int(prediction_old(name,engine,mileage)[0]))

        return result

# Prediction function to predit car price
def prediction_old(name,engine,mileage):
    # Put the user input into an array
    sample = np.array([[name,engine,mileage]])

    # Scale the input data using the trained scaler
    sample_scaled = scaler_old.transform(sample)

    # Predict the car price using the trained model
    result = np.exp(model_old.predict(sample_scaled))

    return result

# The prediction page
@app.route('/predict_new')
def predict_new():
    flash('Hey, you can use the new model in the same way you use the old model. This model is trained from scratch so that it is better suited to predict the price of your car!', 'success')
    return render_template('predict_new.html')

# The route to calculate the prediction result but not accessed by users
@app.route('/process-data_new', methods = ['POST'])
def process_data_new():
    if request.method == 'POST':
        # Getting the values required for prediction
        brand_name = request.form.get('name')
        name = name_map_new.get(brand_name,'32')
        engine = request.form.get('engine', engine_default_new)
        mileage = request.form.get('mileage', mileage_default_new)

        # Convert engine and mileage to float only if they are not empty strings
        if engine:
            engine = float(engine)
        else:
            engine = engine_default_new  # Set a default value if engine is empty

        if mileage:
            mileage = float(mileage)
        else:
            mileage = mileage_default_new # Set a default value if mileage is empty

        # Calling the prediction function, coverting the result to int for user experience and then to string
        # to display on the website
        result = str(int(prediction_new(name,engine,mileage)[0]))

        return result

# Prediction function to predit car price
def prediction_new(name,engine,mileage):
    # Put the user input into an array
    sample = np.array([[name,engine,mileage]])

    # Scale the input data using the trained scaler and add intercepts
    sample_scaled = scaler_new.transform(sample)
    intercept = np.ones((sample_scaled.shape[0], 1))
    sample_scaled   = np.concatenate((intercept, sample_scaled), axis=1)

    # Predict the car price using the trained model
    result = np.exp(model_new.predict(sample_scaled))

    return result

port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)