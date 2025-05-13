import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app 
flask_app = Flask(__name__)

model = pickle.load(open("app/model/best_model.pkl", "rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods = ["POST"])
def predict():
    try:
        hs = float(request.form['hours_studied'])
        sl = float(request.form['hours_slept'])
        at = float(request.form['attendance'])

        input_data = np.array([[hs, sl, at]])
        prediction = model.predict(input_data)[0]

        return render_template("results.html", prediction=round(prediction, 2))
    except Exception as e:
        return render_template("results.html", prediction=f"Error: {e}")
    
  
if __name__ == "__main__":
    flask_app.run(debug=True)
