from flask import Flask, render_template, request
import numpy as np
import joblib as pickle

app = Flask(__name__, template_folder='template', static_folder='static')

model = pickle.load(open('students_marks_predictor_model.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')

model = pickle.load(open('students_marks_predictor_model.pkl','rb'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # form uses name="hours" in the template
        hours = request.form.get('hours', None)
        try:
            hours_val = float(hours)
        except (TypeError, ValueError):
            return render_template('index.html', prediction=None, error='Please enter a valid number for hours')

        # model expects a 2D array: shape (1, n_features)
        features = np.array([[hours_val]])
        prediction = model.predict(features)
        return render_template('index.html', prediction=round(float(prediction[0]), 2))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
