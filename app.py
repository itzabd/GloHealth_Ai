from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from supabase import create_client, Client
import os
from datetime import datetime

app = Flask(__name__)

# Load model artifacts
model = joblib.load('results/production_model.joblib')
le = joblib.load('results/label_encoder.joblib')
feature_cols = joblib.load('results/feature_columns.joblib')

# Supabase configuration
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Supabase table names
PREDICTIONS_TABLE = "predictions"
USERS_TABLE = "auth.users"  # Supabase built-in table
LOCATION_INSIGHTS_TABLE = "location_insights"


def organize_symptoms(features):
    """Organize symptoms into logical groups."""
    groups = defaultdict(list)
    category_map = {
        'fever': 'General', 'fatigue': 'General',
        'cough': 'Respiratory', 'breath': 'Respiratory',
        'vomit': 'Gastrointestinal', 'diarrhoea': 'Gastrointestinal',
        'headache': 'Neurological', 'dizziness': 'Neurological',
        'rash': 'Dermatological', 'itching': 'Dermatological'
    }

    for symptom in features:
        matched = False
        for key, category in category_map.items():
            if key in symptom.lower():
                groups[category].append(symptom)
                matched = True
                break
        if not matched:
            groups['Other'].append(symptom)

    return dict(groups)


def store_prediction(user_id, symptoms, predictions, location_data):
    """Store prediction results in Supabase."""
    try:
        data = {
            "user_id": user_id,
            "symptoms": symptoms,
            "top_prediction": predictions[0]['disease'],
            "confidence": predictions[0]['confidence'],
            "zip_code": location_data.get('zip'),
            "timestamp": datetime.now().isoformat(),
            "full_results": predictions
        }
        supabase.table(PREDICTIONS_TABLE).insert(data).execute()
    except Exception as e:
        app.logger.error(f"Failed to store prediction: {str(e)}")


def update_location_insights(zip_code, disease):
    """Update disease frequency by location."""
    try:
        # Get existing record or initialize
        res = supabase.table(LOCATION_INSIGHTS_TABLE) \
            .select("*") \
            .eq("zip_code", zip_code) \
            .eq("disease", disease) \
            .execute()

        if res.data:
            # Increment count
            supabase.table(LOCATION_INSIGHTS_TABLE) \
                .update({"count": res.data[0]['count'] + 1}) \
                .eq("id", res.data[0]['id']) \
                .execute()
        else:
            # Create new entry
            supabase.table(LOCATION_INSIGHTS_TABLE) \
                .insert({
                "zip_code": zip_code,
                "disease": disease,
                "count": 1,
                "last_updated": datetime.now().isoformat()
            }).execute()
    except Exception as e:
        app.logger.error(f"Failed to update location insights: {str(e)}")


@app.route('/')
def home():
    symptom_groups = organize_symptoms(feature_cols)
    return render_template('index.html', symptom_groups=symptom_groups)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms = data.get('symptoms', {})
        location = {
            'zip': data.get('zipcode'),
            'lat': data.get('lat'),
            'long': data.get('long')
        }

        # Prepare input features
        input_data = {col: 0 for col in feature_cols}
        input_data.update({
            k: 1 for k, v in symptoms.items()
            if k in feature_cols and v in (True, 'true', '1', 1)
        })

        # Make prediction
        probas = model.predict_proba(pd.DataFrame([input_data]))[0]
        top3_idx = np.argsort(probas)[-3:][::-1]
        predictions = [{
            'disease': le.inverse_transform([idx])[0],
            'confidence': float(probas[idx]),
            'probability': f"{probas[idx]:.1%}"
        } for idx in top3_idx]

        # Store data if location provided
        if location.get('zip'):
            store_prediction(
                user_id=data.get('user_id', 'anonymous'),
                symptoms=list(symptoms.keys()),
                predictions=predictions,
                location_data=location
            )
            update_location_insights(location['zip'], predictions[0]['disease'])

        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_prediction': predictions[0]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)