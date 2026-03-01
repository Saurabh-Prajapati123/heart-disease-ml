from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)


model = joblib.load("heart_disease_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)

    result = "Heart Disease Detected 💔" if prediction[0] == 1 else "No Heart Disease ❤️"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)