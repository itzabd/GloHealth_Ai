from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from supabase_client import save_prediction  # We'll create this next
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load models
try:
    model = joblib.load('model_stage1.pkl')
    le = joblib.load('label_encoder.pkl')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    model = None
    le = None

# Get symptom columns from dataset
if os.path.exists('data/X_train.csv'):
    sample_df = pd.read_csv('data/X_train.csv', nrows=1)
    SYMPTOMS = sample_df.columns.tolist()
else:
    SYMPTOMS = []


@app.route('/')
def home():
    return render_template('index.html', symptoms=SYMPTOMS)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or le is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Get symptoms from form
        symptoms_data = request.json.get('symptoms', {})

        # Create input vector
        input_vector = np.zeros((1, len(SYMPTOMS)))
        for i, symptom in enumerate(SYMPTOMS):
            input_vector[0, i] = symptoms_data.get(symptom, 0)

        # Predict
        probas = model.predict_proba(input_vector)[0]
        top3_idx = probas.argsort()[-3:][::-1]
        top3_diseases = le.inverse_transform(top3_idx)
        top3_conf = [round(float(conf), 4) for conf in probas[top3_idx]]

        # Format predictions
        predictions = [
            {"disease": d, "confidence": c}
            for d, c in zip(top3_diseases, top3_conf)
        ]

        # Save to Supabase if configured
        if os.getenv('SUPABASE_URL'):
            try:
                save_prediction(symptoms_data, predictions)
            except Exception as e:
                print(f"Supabase save error: {str(e)}")

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)