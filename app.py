import os
from collections import defaultdict
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, request, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from supabase import create_client
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY') or 'dev-secret-key-change-me'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Supabase configuration
SUPABASE_URL = "https://qmktyfkebpjtihxmfbgp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFta3R5ZmtlYnBqdGloeG1mYmdwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM2NzA3NTYsImV4cCI6MjA2OTI0Njc1Nn0.cAqokDfgN3PgHTQzyW-bPELgJlm3--a-O_Q97SFeTEk"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load ML model
model = joblib.load('results/production_model.joblib')
le = joblib.load('results/label_encoder.joblib')
feature_cols = joblib.load('results/feature_columns.joblib')


# User class
class User(UserMixin):
    def __init__(self, id, email, name, address):
        self.id = id
        self.email = email
        self.name = name
        self.address = address


# User loader
@login_manager.user_loader
def load_user(user_id):
    try:
        # Get the stored access token from session
        access_token = session.get('supabase_access_token')
        if not access_token:
            return None

        # Set the auth header for this request
        supabase.postgrest.auth(access_token)

        # Get user data
        user_response = supabase.auth.get_user(access_token)
        if not user_response.user:
            return None

        # Get profile data
        profile = supabase.from_('user_profiles') \
            .select('*') \
            .eq('id', user_id) \
            .maybe_single() \
            .execute()

        return User(
            id=user_id,
            email=user_response.user.email,
            name=user_response.user.user_metadata.get('name', ''),
            address=profile.data if profile.data else {}
        )
    except Exception as e:
        print(f"Error loading user: {str(e)}")
        return None

# Custom datetime filter
@app.template_filter('datetimeformat')
def format_datetime(value, format="%Y-%m-%d %H:%M"):
    if value is None:
        return ""
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z").strftime(format)
    except:
        return value  # fallback to raw value


# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            response = supabase.auth.sign_in_with_password({
                "email": request.form['email'],
                "password": request.form['password']
            })

            if response.session:
                session.update({
                    'supabase_access_token': response.session.access_token,
                    'supabase_refresh_token': response.session.refresh_token
                })

                user = load_user(response.user.id)
                if user:
                    login_user(user)
                    return redirect(url_for('dashboard'))

        except Exception as e:
            print(f"Login error: {str(e)}")
            return render_template('auth/login.html', error="Login failed")

    return render_template('auth/login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            auth_response = supabase.auth.sign_up({
                "email": request.form['email'],
                "password": request.form['password'],
                "options": {
                    "data": {
                        "name": request.form['name'],
                        "division": request.form['division']
                    },
                    "email_confirm": False
                }
            })

            if auth_response.user:
                # Insert profile data
                profile_response = supabase.from_("user_profiles").insert({
                    "id": auth_response.user.id,
                    "address_line1": request.form['address_line1'],
                    "city": request.form['city'],
                    "division": request.form['division'],
                    "postal_code": request.form['postal_code']
                }).execute()

                # Sign in the user
                response = supabase.auth.sign_in_with_password({
                    "email": request.form['email'],
                    "password": request.form['password']
                })

                session.update({
                    'supabase_access_token': response.session.access_token,
                    'supabase_refresh_token': response.session.refresh_token
                })

                user = load_user(response.user.id)
                if user:
                    login_user(user)
                    return redirect(url_for('dashboard'))

        except Exception as e:
            print(f"Signup error: {str(e)}")
            return render_template('auth/signup.html', error=str(e))

    return render_template('auth/signup.html')


@app.route('/dashboard')
@login_required
def dashboard():
    try:
        predictions = supabase.from_('predictions') \
            .select('*') \
            .eq('user_id', current_user.id) \
            .order('timestamp', desc=True) \
            .limit(5) \
            .execute()

        return render_template('dashboard.html',
                               current_user=current_user,
                               predictions=predictions.data)
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return redirect(url_for('login'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))


# Prediction routes
@app.route('/prediction')
@login_required
def prediction():
    symptom_groups = defaultdict(list)
    category_map = {
        'fever': 'General', 'fatigue': 'General',
        'cough': 'Respiratory', 'breath': 'Respiratory',
        'vomit': 'Gastrointestinal', 'diarrhoea': 'Gastrointestinal',
        'headache': 'Neurological', 'dizziness': 'Neurological',
        'rash': 'Dermatological', 'itching': 'Dermatological'
    }

    for symptom in feature_cols:
        matched = False
        for key, category in category_map.items():
            if key in symptom.lower():
                symptom_groups[category].append(symptom)
                matched = True
                break
        if not matched:
            symptom_groups['Other'].append(symptom)

    return render_template('prediction.html',
                           symptom_groups=symptom_groups,
                           divisions=['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna',
                                      'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh'])


def update_location_insights(division, disease, confidence, zip_code=None, lat=None, long=None):
    """Updates disease prevalence data for a location with proper constraints"""
    try:
        # Get existing record if available
        existing = supabase.from_('location_insights') \
            .select('id, case_count, confidence_score') \
            .eq('division', division) \
            .eq('disease', disease) \
            .maybe_single() \
            .execute()

        # Initialize with defaults
        update_data = {
            "division": division,
            "disease": disease,
            "confidence_score": confidence,
            "last_updated": datetime.now().isoformat(),
            "zip_code": zip_code or "0000"
        }

        if lat and long:
            update_data.update({
                "latitude": lat,
                "longitude": long
            })

        # Calculate running averages if record exists
        if existing and existing.data:
            old_count = existing.data.get('case_count', 1) or 1
            old_conf = existing.data.get('confidence_score', 0) or 0

            update_data.update({
                "case_count": old_count + 1,
                "confidence_score": (old_conf * old_count + confidence) / (old_count + 1),
                "prevalence_score": min(1.0, (old_count + 1) / 1000)
            })

            # Update existing record by ID
            supabase.from_('location_insights') \
                .update(update_data) \
                .eq('id', existing.data['id']) \
                .execute()
        else:
            # Insert new record
            update_data.update({
                "case_count": 1,
                "prevalence_score": 0.001
            })
            supabase.from_('location_insights') \
                .insert(update_data) \
                .execute()

    except Exception as e:
        print(f"Error updating location insights: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # 1. Get input data
        data = request.json
        symptoms = data.get('symptoms', [])
        division = data.get('division')
        lat = data.get('lat')
        long = data.get('long')

        if not symptoms or not division:
            return jsonify({'success': False, 'error': 'Missing symptoms or division'}), 400

        # 2. Get user profile for zip code
        profile = supabase.from_('user_profiles') \
            .select('postal_code') \
            .eq('id', current_user.id) \
            .maybe_single() \
            .execute()
        zip_code = profile.data.get('postal_code') if hasattr(profile, 'data') and profile.data else None

        # 3. Prepare model input
        input_data = {col: 0 for col in feature_cols}
        input_data.update({sym: 1 for sym in symptoms if sym in feature_cols})

        # 4. Get base probabilities
        probas = model.predict_proba(pd.DataFrame([input_data]))[0]

        # 5. Apply location boosts
        boosted_probas = []
        for idx, base_prob in enumerate(probas):
            disease = le.inverse_transform([idx])[0]
            boost_factor = calculate_location_boost(division, disease)
            boosted_probas.append(base_prob * boost_factor)

        # 6. Normalize probabilities
        total = sum(boosted_probas)
        normalized_probas = [p / total for p in boosted_probas] if total > 0 else boosted_probas

        # 7. Prepare predictions
        top3_idx = np.argsort(normalized_probas)[-3:][::-1]
        predictions = [{
            'disease': le.inverse_transform([idx])[0],
            'confidence': float(normalized_probas[idx]),
            'probability': f"{normalized_probas[idx]:.1%}",
            'regional_influence': calculate_location_boost(division, le.inverse_transform([idx])[0]) - 1
        } for idx in top3_idx]

        # 8. Save results
        prediction_data = {
            "user_id": current_user.id,
            "symptoms": symptoms,
            "top_prediction": predictions[0]['disease'],
            "confidence": predictions[0]['confidence'],
            "zip_code": zip_code,
            "division": division,
            "latitude": lat,
            "longitude": long,
            "full_results": predictions
        }
        supabase.from_('predictions').insert(prediction_data).execute()

        # 9. Update location insights
        update_location_insights(
            division=division,
            disease=predictions[0]['disease'],
            confidence=predictions[0]['confidence'],
            zip_code=zip_code,
            lat=lat,
            long=long
        )

        return jsonify({
            'success': True,
            'predictions': predictions,
            'location_factors': {
                'division': division,
                'zip_code': zip_code
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


def save_prediction_and_update_insights(user_id, symptoms, predictions, division, lat, long):
    """Save prediction and update regional disease prevalence data."""
    try:
        # Get user's postal code from profile
        profile = supabase.from_('user_profiles') \
            .select('postal_code') \
            .eq('id', user_id) \
            .maybe_single() \
            .execute()

        zip_code = profile.data.get('postal_code') if hasattr(profile, 'data') and profile.data else None

        # 1. Save the prediction
        prediction_data = {
            "user_id": user_id,
            "symptoms": symptoms,
            "top_prediction": predictions[0]['disease'],
            "confidence": predictions[0]['confidence'],
            "division": division,
            "latitude": lat,
            "longitude": long,
            "full_results": predictions,
            "zip_code": zip_code  # Add zip_code to prediction
        }

        prediction_result = supabase.from_('predictions').insert(prediction_data).execute()

        if not hasattr(prediction_result, 'data'):
            raise Exception("Failed to save prediction")

        # 2. Update location insights
        top_pred = predictions[0]
        update_data = {
            "zip_code": zip_code or "0000",  # Default if null
            "division": division,
            "disease": top_pred['disease'],
            "confidence_score": top_pred['confidence'],
            "last_updated": datetime.now().isoformat()
        }

        # Only add coordinates if available
        if lat and long:
            update_data.update({
                "latitude": lat,
                "longitude": long
            })

        # Check if record exists
        existing_resp = supabase.from_('location_insights') \
            .select('*') \
            .eq('division', division) \
            .eq('disease', top_pred['disease']) \
            .maybe_single() \
            .execute()

        if hasattr(existing_resp, 'data') and existing_resp.data:
            # Update existing record
            old_conf = existing_resp.data.get('confidence_score', 0)
            update_data['confidence_score'] = (old_conf * 0.7) + (top_pred['confidence'] * 0.3)

            update_result = supabase.from_('location_insights') \
                .update(update_data) \
                .eq('id', existing_resp.data['id']) \
                .execute()
        else:
            # Create new record
            update_result = supabase.from_('location_insights') \
                .insert(update_data) \
                .execute()

        if not hasattr(update_result, 'data'):
            raise Exception("Failed to update location insights")

    except Exception as e:
        print(f"Error in save_prediction_and_update_insights: {str(e)}")
        raise

def _get_seasonal_adjustment(disease):
    """Get seasonal adjustment factor for a disease."""
    try:
        month = datetime.now().month
        seasonal_data = supabase.from_('disease_seasonality') \
            .select('avg_cases') \
            .eq('disease', disease) \
            .eq('month', month) \
            .maybe_single() \
            .execute()

        if hasattr(seasonal_data, 'data') and seasonal_data.data:
            avg_cases = seasonal_data.data.get('avg_cases', 1.0)
            return 0.8 + (0.4 * avg_cases)
    except Exception as e:
        print(f"Error getting seasonal data: {str(e)}")

    return 1.0  # Default no adjustment if no data or error


def calculate_location_boost(division: str, disease: str) -> float:
    """
    Calculate location-based probability boost using actual prevalence data
    Returns: Boost multiplier between 1.0 (no boost) and 2.0 (max boost)
    """
    try:
        # Fetch latest disease prevalence for the division
        res = supabase.from_('location_insights') \
            .select('confidence_score, prevalence_score') \
            .eq('division', division) \
            .eq('disease', disease) \
            .order('last_updated', desc=True) \
            .limit(1) \
            .execute()

        if res.data and len(res.data) > 0:
            record = res.data[0]
            # Calculate boost using both confidence and prevalence
            raw_boost = 1.0 + (record['confidence_score'] or 0) * (record['prevalence_score'] or 0.5)
            return min(2.0, max(1.0, raw_boost))  # Clamp between 1.0-2.0

    except Exception as e:
        print(f"Error calculating location boost: {str(e)}")

    return 1.0  # Default no boost

if __name__ == '__main__':
    app.run(debug=True)