from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict

app = Flask(__name__)

# Load model artifacts
try:
    model = joblib.load('results/disease_predictor.joblib')
    le = joblib.load('results/label_encoder.joblib')
    feature_cols = joblib.load('results/feature_columns.joblib')
    print("Successfully loaded all model artifacts")
except Exception as e:
    print(f"Error loading model artifacts: {str(e)}")
    raise


def organize_symptoms(features):
    """Organize symptoms into logical groups."""
    groups = {
        'General': [],
        'Respiratory': [],
        'Gastrointestinal': [],
        'Neurological': [],
        'Dermatological': [],
        'Other': []
    }

    # Categorize each symptom
    for symptom in features:
        symptom_lower = symptom.lower()
        if 'fever' in symptom_lower or 'fatigue' in symptom_lower:
            groups['General'].append(symptom)
        elif 'cough' in symptom_lower or 'breath' in symptom_lower:
            groups['Respiratory'].append(symptom)
        elif 'vomit' in symptom_lower or 'diarrhoea' in symptom_lower or 'abdominal' in symptom_lower:
            groups['Gastrointestinal'].append(symptom)
        elif 'headache' in symptom_lower or 'dizziness' in symptom_lower or 'vertigo' in symptom_lower:
            groups['Neurological'].append(symptom)
        elif 'rash' in symptom_lower or 'itching' in symptom_lower or 'skin' in symptom_lower:
            groups['Dermatological'].append(symptom)
        else:
            groups['Other'].append(symptom)

    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


@app.route('/')
def home():
    """Render the symptom checker interface."""
    symptom_groups = organize_symptoms(feature_cols)
    return render_template('index.html', symptom_groups=symptom_groups)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions."""
    try:
        symptoms = request.json.get('symptoms', {})
        input_data = {col: 0 for col in feature_cols}
        input_data.update({
            k: 1 for k, v in symptoms.items()
            if k in feature_cols and (v == True or v == 'true' or v == '1')
        })
        input_df = pd.DataFrame([input_data])[feature_cols]

        probas = model.predict_proba(input_df)[0]
        top3_idx = np.argsort(probas)[-3:][::-1]

        results = [{
            'disease': le.inverse_transform([idx])[0],
            'confidence': float(probas[idx]),
            'probability': f"{probas[idx]:.1%}"
        } for idx in top3_idx]

        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': results[0]['disease']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)