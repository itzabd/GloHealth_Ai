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


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        division = data.get('division')
        lat = data.get('lat')
        long = data.get('long')

        if not symptoms or not division:
            return jsonify({'success': False, 'error': 'Missing symptoms or division'}), 400

        # Prepare model input
        input_data = {col: 0 for col in feature_cols}
        input_data.update({sym: 1 for sym in symptoms if sym in feature_cols})

        # Get model probabilities
        probas = model.predict_proba(pd.DataFrame([input_data]))[0]

        # Fetch location stats for bias
        loc_stats_resp = supabase.from_('location_insights') \
            .select('disease, confidence_score') \
            .eq('division', division) \
            .execute()

        bias_map = {}
        if loc_stats_resp.data:
            bias_map = {row['disease']: row['confidence_score'] or 0
                        for row in loc_stats_resp.data}

        # Apply bias
        biased_probas = []
        for idx, base_prob in enumerate(probas):
            disease = le.inverse_transform([idx])[0]
            if disease in bias_map:
                bias_factor = 1 + bias_map[disease]  # adjust multiplier as needed
                biased_probas.append(base_prob * bias_factor)
            else:
                biased_probas.append(base_prob)

        # Normalize probabilities
        total = sum(biased_probas)
        if total > 0:
            biased_probas = [p / total for p in biased_probas]

        # Get top 3 predictions
        top3_idx = np.argsort(biased_probas)[-3:][::-1]
        predictions = [{
            'disease': le.inverse_transform([idx])[0],
            'confidence': float(biased_probas[idx]),
            'probability': f"{biased_probas[idx]:.1%}"
        } for idx in top3_idx]

        # Save to predictions table
        supabase.from_('predictions').insert({
            "user_id": current_user.id,
            "symptoms": symptoms,
            "top_prediction": predictions[0]['disease'],
            "confidence": predictions[0]['confidence'],
            "division": division,
            "latitude": lat,
            "longitude": long,
            "full_results": predictions
        }).execute()

        # Update location_insights with top disease
        top_pred = predictions[0]
        existing_resp = supabase.from_('location_insights') \
            .select('*') \
            .eq('division', division) \
            .eq('disease', top_pred['disease']) \
            .maybe_single() \
            .execute()

        if existing_resp.data is not None:
            old_conf = existing_resp.data.get('confidence_score') or 0
            new_conf = (old_conf + top_pred['confidence']) / 2
            supabase.from_('location_insights').update({
                "confidence_score": new_conf,
                "last_updated": datetime.now().isoformat()
            }).eq('id', existing_resp.data['id']).execute()
        else:
            supabase.from_('location_insights').insert({
                "division": division,
                "disease": top_pred['disease'],
                "confidence_score": top_pred['confidence']
            }).execute()

        return jsonify({'success': True, 'predictions': predictions})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)