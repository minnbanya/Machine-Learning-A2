from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    title = "Home page"
    name = "Would you like to predict how much your car would sell for?"
    return render_template('index.html', title=title, name=name)

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        #Getting the values required for prediction
        name = request.form.get(loaded_data['name_map']['name'],'32')

if __name__ == '__main__':
    app.run(debug=True)