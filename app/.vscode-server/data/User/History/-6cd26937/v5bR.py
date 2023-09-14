from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

filename = 'model/car_price.model'
loaded_data = pickle.load(open(filename, 'rb'))

model = loaded_data['model']
scaler = loaded_data['scaler']
name_map = loaded_data['name_map']
engine_default = loaded_data['engine_default']
mileage_default = loaded_data['mileage_default']

@app.route('/')
def index():
    title = "Home page"
    name = "Would you like to predict how much your car would sell for?"
    return render_template('index.html', title=title, name=name)

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/process_data', methods = ['POST'])
def process_data():
    if request.method == 'POST':
        #Getting the values required for prediction
        name = request.form.get(name_map['name'],'32')
        engine = request.form.get('engine', engine_default)
        mileage = request.form.get('mileage', mileage_default)

        result = prediction(name,engine,mileage)

        return result

def prediction(name,engine,mileage):
    sample = np.array([[name,engine,mileage]])
    sample_scaled = scaler.transform(sample)
    result = np.exp(model.predict(sample_scaled))

    return result

if __name__ == '__main__':
    app.run(debug=True)