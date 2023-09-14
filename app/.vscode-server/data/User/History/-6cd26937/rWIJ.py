from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    title = "Welcome to Car Price Prediction!"
    name = "Would you like to predict how much your car would sell for?"
    return render_template('index.html', title=title, name=name)

if __name__ == '__main__':
    app.run(debug=True)