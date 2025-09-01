import os
from collections import defaultdict
from flask import Flask, render_template, redirect, url_for, request, session, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from supabase import create_client
import joblib
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from flask import Flask, render_template, redirect, url_for, request, flash
from datetime import datetime, timedelta
from flask import abort, flash, redirect, url_for, render_template
from functools import wraps
from flask import abort
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
    def __init__(self, id, email, name, address, is_admin=False):
        self.id = id
        self.email = email
        self.name = name
        self.address = address
        self.is_admin = is_admin  # New field

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
        profile_resp = supabase.from_('user_profiles') \
            .select('*') \
            .eq('id', user_id) \
            .maybe_single() \
            .execute()
        profile = profile_resp.data if hasattr(profile_resp, 'data') else {}

        # Return User instance with is_admin
        return User(
            id=user_id,
            email=user_response.user.email,
            name=user_response.user.user_metadata.get('name', ''),
            address=profile,
            is_admin=profile.get('is_admin', False)  # Fetch admin flag
        )

    except Exception as e:
        print(f"Error loading user: {str(e)}")
        return None
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not getattr(current_user, 'is_admin', False):
            abort(403)  # Forbidden
        return f(*args, **kwargs)
    return decorated_function
@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    # Example: show system stats, user management, etc.
    users = supabase.from_('user_profiles').select('*').execute().data
    appointments = supabase.from_('appointments').select('*').execute().data
    return render_template('admin_dashboard.html', users=users, appointments=appointments)


@app.route('/admin/users')
@login_required
def admin_users():
    if not getattr(current_user, 'is_admin', False):
        flash("Unauthorized access", "danger")
        return redirect(url_for('dashboard'))

    try:
        users_resp = supabase.from_("user_profiles").select("*").execute()
        users = users_resp.data if hasattr(users_resp, "data") else []
        return render_template("admin_users.html", users=users)
    except Exception as e:
        print(f"Admin Users error: {e}")
        flash("Error fetching users", "danger")
        return redirect(url_for("admin_dashboard"))

# View all doctors
@app.route("/admin/doctors")
@login_required
def admin_doctors():
    if not getattr(current_user, "is_admin", False):
        flash("Unauthorized access", "danger")
        return redirect(url_for("dashboard"))

    try:
        resp = supabase.from_("doctors").select("*").execute()
        doctors = resp.data if hasattr(resp, "data") else []
        return render_template("admin_doctors.html", doctors=doctors)
    except Exception as e:
        flash(f"Error fetching doctors: {str(e)}", "danger")
        return redirect(url_for("admin_dashboard"))


# Add doctor page
@app.route("/admin/doctors/add", methods=["GET", "POST"])
@login_required
def add_doctor_page():
    if not getattr(current_user, "is_admin", False):
        flash("Unauthorized access", "danger")
        return redirect(url_for("admin_doctors"))

    if request.method == "POST":
        try:
            data = request.form
            supabase.from_("doctors").insert({
                "name": data["name"],
                "specialty": data["specialty"],
                "division": data.get("division"),
                "district": data.get("district"),
                "hospital": data.get("hospital"),
                "consultation_fee": data["consultation_fee"],
                "availability": data.get("availability"),
                "contact": data.get("contact")
            }).execute()
            flash("Doctor added successfully!", "success")
            return redirect(url_for("admin_doctors"))
        except Exception as e:
            flash(f"Error adding doctor: {str(e)}", "danger")

    return render_template("add_doctor.html")


# Edit doctor page
@app.route("/admin/doctors/edit/<doctor_id>", methods=["GET", "POST"])
@login_required
def edit_doctor_page(doctor_id):
    if not getattr(current_user, "is_admin", False):
        flash("Unauthorized access", "danger")
        return redirect(url_for("admin_doctors"))

    try:
        resp = supabase.from_("doctors").select("*").eq("id", doctor_id).single().execute()
        doctor = resp.data
    except Exception as e:
        flash(f"Error fetching doctor: {str(e)}", "danger")
        return redirect(url_for("admin_doctors"))

    if request.method == "POST":
        try:
            data = request.form
            supabase.from_("doctors").update({
                "name": data["name"],
                "specialty": data["specialty"],
                "division": data.get("division"),
                "district": data.get("district"),
                "hospital": data.get("hospital"),
                "consultation_fee": data["consultation_fee"],
                "availability": data.get("availability"),
                "contact": data.get("contact")
            }).eq("id", doctor_id).execute()
            flash("Doctor updated successfully!", "success")
            return redirect(url_for("admin_doctors"))
        except Exception as e:
            flash(f"Error updating doctor: {str(e)}", "danger")

    return render_template("edit_doctor.html", doctor=doctor)


# Delete doctor page
@app.route("/admin/doctors/delete/<doctor_id>", methods=["POST"])
@login_required
def delete_doctor_page(doctor_id):
    if not getattr(current_user, "is_admin", False):
        flash("Unauthorized access", "danger")
        return redirect(url_for("admin_doctors"))

    try:
        supabase.from_("doctors").delete().eq("id", doctor_id).execute()
        flash("Doctor deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting doctor: {str(e)}", "danger")

    return redirect(url_for("admin_doctors"))
# View all appointments
@app.route('/admin/appointments')
@login_required
def admin_appointments():
    if not getattr(current_user, 'is_admin', False):
        flash("Unauthorized access", "danger")
        return redirect(url_for('dashboard'))

    try:
        appointments_resp = supabase.from_('appointments').select('*').execute()
        appointments = appointments_resp.data if hasattr(appointments_resp, 'data') else []

        doctors_resp = supabase.from_('doctors').select('id, name').execute()
        doctors = {d['id']: d['name'] for d in doctors_resp.data} if hasattr(doctors_resp, 'data') else {}

        for a in appointments:
            a['doctor_name'] = doctors.get(a['doctor_id'], 'N/A')

        return render_template('admin_appointments.html', appointments=appointments)
    except Exception as e:
        print(f"Admin Appointments error: {e}")
        flash("Error fetching appointments", "danger")
        return redirect(url_for('admin_dashboard'))


# Add appointment
@app.route("/admin/appointments/add", methods=["GET", "POST"])
@login_required
def add_appointment_page():
    if not getattr(current_user, "is_admin", False):
        flash("Unauthorized access", "danger")
        return redirect(url_for("admin_appointments"))

    # Fetch doctors and users
    doctors_resp = supabase.from_("doctors").select("*").execute()
    doctors = doctors_resp.data if hasattr(doctors_resp, "data") else []

    users_resp = supabase.from_("user_profiles").select("*").execute()
    users = users_resp.data if hasattr(users_resp, "data") else []

    if request.method == "POST":
        try:
            data = request.form
            # Find selected user to get name/email
            selected_user = next((u for u in users if u["id"] == data["user_id"]), None)
            supabase.from_("appointments").insert({
                "user_id": data["user_id"],
                "user_name": selected_user["full_name"] if selected_user else "N/A",
                "user_email": selected_user["email"] if selected_user else "N/A",
                "doctor_id": data["doctor_id"],
                "scheduled_time": data["scheduled_time"],
                "status": data.get("status", "pending"),
                "payment_status": data.get("payment_status", "unpaid")
            }).execute()

            flash("Appointment added successfully!", "success")
            return redirect(url_for("admin_appointments"))
        except Exception as e:
            flash(f"Error adding appointment: {str(e)}", "danger")

    return render_template("add_appointment.html", doctors=doctors, users=users)

# Edit appointment
@app.route("/admin/appointments/edit/<appointment_id>", methods=["GET", "POST"])
@login_required
def edit_appointment_page(appointment_id):
    if not getattr(current_user, "is_admin", False):
        flash("Unauthorized access", "danger")
        return redirect(url_for("admin_appointments"))

    try:
        resp = supabase.from_("appointments").select("*").eq("id", appointment_id).single().execute()
        appointment = resp.data

        doctors_resp = supabase.from_("doctors").select("*").execute()
        doctors = doctors_resp.data if hasattr(doctors_resp, "data") else []

        users_resp = supabase.from_("user_profiles").select("*").execute()
        users = users_resp.data if hasattr(users_resp, "data") else []

    except Exception as e:
        flash(f"Error fetching appointment: {str(e)}", "danger")
        return redirect(url_for("admin_appointments"))

    if request.method == "POST":
        try:
            data = request.form
            selected_user = next((u for u in users if u["id"] == data["user_id"]), None)
            supabase.from_("appointments").update({
                "user_id": data["user_id"],
                "user_name": selected_user["full_name"] if selected_user else "N/A",
                "user_email": selected_user["email"] if selected_user else "N/A",
                "doctor_id": data["doctor_id"],
                "scheduled_time": data["scheduled_time"],
                "status": data.get("status", "pending"),
                "payment_status": data.get("payment_status", "unpaid")
            }).eq("id", appointment_id).execute()

            flash("Appointment updated successfully!", "success")
            return redirect(url_for("admin_appointments"))
        except Exception as e:
            flash(f"Error updating appointment: {str(e)}", "danger")

    return render_template("edit_appointment.html", appointment=appointment, doctors=doctors, users=users)


# Delete appointment
@app.route("/admin/appointments/delete/<appointment_id>", methods=["POST"])
@login_required
def delete_appointment_page(appointment_id):
    if not getattr(current_user, "is_admin", False):
        flash("Unauthorized access", "danger")
        return redirect(url_for("admin_appointments"))

    try:
        supabase.from_("appointments").delete().eq("id", appointment_id).execute()
        flash("Appointment deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting appointment: {str(e)}", "danger")

    return redirect(url_for("admin_appointments"))


@app.route("/edit_user/<user_id>", methods=["GET", "POST"])
def edit_user(user_id):
    if request.method == "POST":
        full_name = request.form["full_name"]
        email = request.form["email"]
        city = request.form["city"]
        division = request.form["division"]

        # Update user_profiles
        update_response = supabase.from_("user_profiles").update({
            "full_name": full_name,
            "email": email,
            "city": city,
            "division": division
        }).eq("id", user_id).execute()

        flash("User updated successfully!", "success")
        return redirect(url_for("admin_users"))

    # Fetch user for pre-fill
    user = supabase.from_("user_profiles").select("*").eq("id", user_id).single().execute()
    return render_template("edit_user.html", user=user.data)



@app.route("/delete_user/<user_id>", methods=["GET"])
def delete_user(user_id):
    try:
        # First delete from user_profiles
        supabase.from_("user_profiles").delete().eq("id", user_id).execute()

        # # Then delete from auth.users
        # supabase.auth.admin.delete_user(user_id)

        flash("User deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting user: {str(e)}", "danger")

    return redirect(url_for("admin_users"))

# Add Doctor

# Edit Appointment
@app.route('/admin/appointments/edit/<appointment_id>', methods=['GET', 'POST'])
@login_required
def edit_appointment(appointment_id):
    if not getattr(current_user, 'is_admin', False):
        flash("Unauthorized access", "danger")
        return redirect(url_for('dashboard'))

    try:
        resp = supabase.from_('appointments').select('*').eq('id', appointment_id).maybe_single().execute()
        appointment = resp.data

        if request.method == 'POST':
            updated_data = {
                "scheduled_time": request.form.get("scheduled_time"),
                "status": request.form.get("status"),
                "payment_status": request.form.get("payment_status")
            }
            supabase.from_('appointments').update(updated_data).eq('id', appointment_id).execute()
            flash("Appointment updated successfully", "success")
            return redirect(url_for('admin_appointments'))

        return render_template('edit_appointment.html', appointment=appointment)
    except Exception as e:
        print(f"Edit appointment error: {str(e)}")
        flash("Error editing appointment", "danger")
        return redirect(url_for('admin_appointments'))


# Delete Appointment
@app.route('/admin/appointments/delete/<appointment_id>', methods=['POST'])
@login_required
def delete_appointment(appointment_id):
    if not getattr(current_user, 'is_admin', False):
        flash("Unauthorized access", "danger")
        return redirect(url_for('dashboard'))

    try:
        supabase.from_('appointments').delete().eq('id', appointment_id).execute()
        flash("Appointment deleted successfully", "success")
    except Exception as e:
        print(f"Delete appointment error: {str(e)}")
        flash("Error deleting appointment", "danger")
    return redirect(url_for('admin_appointments'))
@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
def admin_settings():
    if not getattr(current_user, 'is_admin', False):
        flash("Unauthorized access", "danger")
        return redirect(url_for('dashboard'))

    try:
        # Fetch current settings
        resp = supabase.from_('system_settings').select('*').maybe_single().execute()
        settings = resp.data or {}

        if request.method == 'POST':
            updated_settings = {
                "site_name": request.form.get("site_name"),
                "support_email": request.form.get("support_email"),
                "checkup_fee": request.form.get("checkup_fee")
            }
            if settings.get("id"):
                supabase.from_('system_settings').update(updated_settings).eq('id', settings["id"]).execute()
            else:
                supabase.from_('system_settings').insert(updated_settings).execute()

            flash("Settings updated successfully", "success")
            return redirect(url_for('admin_settings'))

        return render_template('admin_settings.html', settings=settings)
    except Exception as e:
        print(f"Update settings error: {str(e)}")
        flash("Error updating settings", "danger")
        return redirect(url_for('admin_dashboard'))



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
                        "full_name": request.form['name'],  # 👈 use full_name consistently
                        "division": request.form['division']
                    },
                    "email_confirm": False
                }
            })

            if auth_response.user:
                user = auth_response.user  # shortcut

                profile_response = supabase.from_("user_profiles").insert({
                    "id": user.id,
                    "address_line1": request.form['address_line1'],
                    "city": request.form['city'],
                    "division": request.form['division'],
                    "postal_code": request.form['postal_code'],
                    "full_name": user.user_metadata.get("full_name"),  # 👈 now matches
                    "email": user.email
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
        # Fetch last 5 predictions
        predictions_resp = supabase.from_('predictions') \
            .select('*') \
            .eq('user_id', str(current_user.id)) \
            .order('timestamp', desc=True) \
            .limit(5) \
            .execute()
        predictions = predictions_resp.data if hasattr(predictions_resp, 'data') else []

        # Fetch all user appointments
        appointments_resp = supabase.from_('appointments') \
            .select('*') \
            .eq('user_id', str(current_user.id)) \
            .order('scheduled_time', desc=True) \
            .execute()
        appointments = appointments_resp.data if hasattr(appointments_resp, 'data') else []

        return render_template('dashboard.html',
                               current_user=current_user,
                               predictions=predictions,
                               appointments=appointments)
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return redirect(url_for('login'))



@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))


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

    # Fetch active subscription for current user from Supabase
    response = supabase.table("user_subscriptions") \
                       .select("plan_name") \
                       .eq("user_id", str(current_user.id)) \
                       .eq("active", True) \
                       .order("start_date", desc=True) \
                       .limit(1) \
                       .execute()

    user_plan_name = response.data[0]['plan_name'] if response.data else None

    return render_template(
        'prediction.html',
        symptom_groups=symptom_groups,
        divisions=['Dhaka', 'Chittagong', 'Rajshahi', 'Khulna', 'Barisal', 'Sylhet', 'Rangpur', 'Mymensingh'],
        user_plan_name=user_plan_name  # Pass the active plan to the template
    )

def update_location_insights(division, disease, confidence, zip_code=None, lat=None, long=None):
    """Insert a new location insight entry every time"""
    try:
        # Prepare data
        insert_data = {
            "division": division,
            "disease": disease,
            "confidence_score": float(confidence),
            "last_updated": datetime.now().isoformat(),
            "zip_code": zip_code or "0000",
            "case_count": 1,
            "prevalence_score": 0.001  # initial placeholder
        }

        # Add coordinates if available
        if lat and long:
            insert_data.update({
                "lat": float(lat),
                "lon": float(long),
                "latitude": float(lat),
                "longitude": float(long)
            })

        # Insert new record
        supabase.from_('location_insights').insert(insert_data).execute()

    except Exception as e:
        print(f"⚠️ Error inserting location insight: {str(e)}")
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
        zip_code = profile.data.get('postal_code') if profile and getattr(profile, 'data', None) else None

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
        normalized_probas = [p / total for p in boosted_probas] if total > 0 else [0]*len(boosted_probas)

        # 7. Prepare predictions
        top3_idx = np.argsort(normalized_probas)[-3:][::-1]
        predictions = [{
            'disease': le.inverse_transform([idx])[0],
            'confidence': float(normalized_probas[idx]),
            'probability': f"{normalized_probas[idx]*100:.1f}%",
            'regional_influence': calculate_location_boost(division, le.inverse_transform([idx])[0]) - 1
        } for idx in top3_idx]

        # 8. Save prediction results
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

        # 9. Insert new location insight every time
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
            .select('confidence_score, prevalence_score, case_count') \
            .eq('division', division) \
            .eq('disease', disease) \
            .order('last_updated', desc=True) \
            .limit(1) \
            .execute()

        if res.data and len(res.data) > 0:
            record = res.data[0]
            # Calculate boost using confidence, prevalence, and case count
            conf_score = record.get('confidence_score', 0) or 0
            prev_score = record.get('prevalence_score', 0) or 0
            case_count = record.get('case_count', 0) or 0

            # More sophisticated boost calculation
            raw_boost = 1.0 + (conf_score * 0.6) + (prev_score * 0.3) + (min(case_count, 100) / 100 * 0.1)
            return min(2.0, max(1.0, raw_boost))  # Clamp between 1.0-2.0

    except Exception as e:
        print(f"Error calculating location boost: {str(e)}")

    return 1.0  # Default no boost
@app.route('/geo_insights')
@login_required
def geo_insights():
    try:
        # GET DATA FROM location_insights TABLE
        response = supabase.from_('location_insights') \
            .select('*') \
            .order('last_updated', desc=True) \
            .execute()

        if not hasattr(response, 'data') or not response.data:
            return render_template('geo_insights.html',
                                   current_user=current_user,
                                   error="No location insights data available")

        df = pd.DataFrame(response.data)

        # Create Bangladesh map
        bd_center = [23.6850, 90.3563]
        bd_map = folium.Map(location=bd_center, zoom_start=7, tiles='cartodbpositron')

        # Bangladesh boundary
        bd_bounds = [[20.5, 88.0], [26.5, 92.5]]
        folium.Rectangle(
            bounds=bd_bounds,
            color='#000000',
            weight=2,
            fill=True,
            fillColor='#ffff00',
            fillOpacity=0.1,
            popup='Bangladesh'
        ).add_to(bd_map)

        # Division coordinates (approximate centers)
        division_coordinates = {
            'Dhaka': [23.8103, 90.4125],
            'Chittagong': [22.3569, 91.7832],
            'Rajshahi': [24.3745, 88.6042],
            'Khulna': [22.8456, 89.5403],
            'Barisal': [22.7010, 90.3535],
            'Sylhet': [24.8910, 91.8710],
            'Rangpur': [25.7439, 89.2752],
            'Mymensingh': [24.7471, 90.4203]
        }

        # Heatmap and marker cluster
        heat_data = []
        marker_cluster = MarkerCluster(name="Cases").add_to(bd_map)

        for _, row in df.iterrows():
            division = row.get('division')
            if division and division in division_coordinates:
                lat, lon = division_coordinates[division]
                case_count = row.get('case_count', 1)
                if pd.notna(lat) and pd.notna(lon) and pd.notna(case_count):
                    heat_data.append([float(lat), float(lon), float(case_count)])
                    popup = f"""
                    <b>{row.get('disease', 'N/A')}</b><br>
                    Cases: {case_count}<br>
                    Division: {division}<br>
                    Confidence: {row.get('confidence_score', 0):.1%}
                    """
                    folium.Marker(
                        location=[float(lat), float(lon)],
                        popup=popup,
                        icon=folium.Icon(
                            color='red' if case_count > 10
                            else 'orange' if case_count > 5
                            else 'green'
                        )
                    ).add_to(marker_cluster)

        if heat_data:
            HeatMap(
                heat_data,
                name="Case Density",
                radius=25,
                blur=20,
                max_zoom=1,
                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
            ).add_to(bd_map)

        folium.LayerControl().add_to(bd_map)
        map_html = bd_map._repr_html_() if bd_map else None

        # Prepare statistics
        division_counts_dict = {}
        disease_totals = {}
        top_diseases = []
        all_diseases = []

        if not df.empty and 'division' in df.columns and 'disease' in df.columns:
            # Pivot table for counts
            division_counts = df.pivot_table(
                index='division',
                columns='disease',
                values='case_count',
                aggfunc='sum',
                fill_value=0
            )
            division_counts_dict = division_counts.to_dict('index')
            disease_totals = division_counts.sum().to_dict()
            top_diseases = division_counts.sum().nlargest(5).index.tolist()
            all_diseases = df['disease'].unique().tolist()

        return render_template('geo_insights.html',
                               current_user=current_user,
                               map_html=map_html,
                               division_counts=division_counts_dict,
                               disease_totals=disease_totals,
                               top_diseases=top_diseases,
                               all_diseases=all_diseases)

    except Exception as e:
        print(f"Geo insights error: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('geo_insights.html',
                               current_user=current_user,
                               error=str(e))

# THEN PUT THE TEMPLATE FILTER OUTSIDE THE FUNCTION
@app.template_filter('get_disease_color')
def get_disease_color_filter(disease):
    """Template filter to get disease color."""
    color_map = {
        'Diabetes': 'red',
        'Hypertension': 'blue',
        'Asthma': 'green',
        'Flu': 'orange',
        'COVID-19': 'purple',
        'Dengue': 'darkred',
        'Malaria': 'pink'
    }
    return color_map.get(disease, 'gray')

# List doctors (paid feature)
@app.route('/doctors')
@login_required
def doctors():
    division = request.args.get('division')
    specialty = request.args.get('specialty')

    query = supabase.from_('doctors').select('*')
    if division:
        query = query.eq('division', division)
    if specialty:
        query = query.eq('specialty', specialty)

    response = query.execute()
    doctors_list = response.data if hasattr(response, 'data') else []

    return render_template('doctors.html', doctors=doctors_list)
# Book appointment route
@app.route('/book_appointment/<doctor_id>', methods=['GET', 'POST'])
@login_required
def book_appointment(doctor_id):
    try:
        # Fetch doctor details
        doctor_resp = supabase.from_('doctors')\
                              .select('*')\
                              .eq('id', doctor_id)\
                              .maybe_single()\
                              .execute()
        doctor = doctor_resp.data if hasattr(doctor_resp, 'data') else None
        if not doctor:
            return "Doctor not found", 404

        # Fetch user's active subscription with checkup points
        sub_resp = supabase.table("user_subscriptions")\
            .select("*")\
            .eq("user_id", str(current_user.id))\
            .eq("active", True)\
            .order("start_date", desc=True)\
            .limit(1)\
            .execute()
        subscription = sub_resp.data[0] if sub_resp.data else None
        if not subscription:
            flash("You need an active subscription to book an appointment.", "danger")
            return redirect(url_for('plans'))

        if request.method == 'POST':
            scheduled_time = request.form.get('scheduled_time')
            payment_done = request.form.get('payment_done') == 'yes'  # checkbox: paid
            use_points = request.form.get('use_points') == 'yes'      # checkbox: free checkup

            if not scheduled_time:
                flash("Please select a date and time", "warning")
                return redirect(url_for('book_appointment', doctor_id=doctor_id))

            # Default status/payment
            status = 'pending'
            payment_status = 'unpaid'

            # Handle free checkup
            if use_points:
                if subscription["checkup_points"] <= 0:
                    flash("No free checkups remaining for this month.", "warning")
                    return redirect(url_for('features', plan_name=subscription["plan_name"]))

                # Deduct one free checkup point
                new_points = subscription["checkup_points"] - 1
                supabase.table("user_subscriptions") \
                    .update({"checkup_points": new_points}) \
                    .eq("id", subscription["id"]) \
                    .execute()

                subscription["checkup_points"] = new_points
                status = 'confirmed'
                payment_status = 'free'
            else:
                # Paid booking
                status = 'confirmed' if payment_done else 'pending'
                payment_status = 'paid' if payment_done else 'unpaid'

            # Insert appointment
            supabase.from_('appointments').insert({
                'user_id': str(current_user.id),
                'doctor_id': doctor_id,
                'scheduled_time': scheduled_time,
                'status': status,
                'payment_status': payment_status
            }).execute()

            flash("Appointment booked successfully!", "success")
            return redirect(url_for('appointments'))

        # GET request: show form
        return render_template(
            'book_appointment.html',
            doctor=doctor,
            subscription=subscription
        )

    except Exception as e:
        print(f"Book appointment error: {str(e)}")
        return "Error booking appointment", 500

@app.route('/appointments')
@login_required
def appointments():
    try:
        # Fetch user's appointments with doctor info
        response = supabase.from_('appointments') \
            .select('*, doctors(*)') \
            .eq('user_id', str(current_user.id)) \
            .order('scheduled_time', desc=True) \
            .execute()

        # Extract data safely
        appointments_list = response.data if hasattr(response, 'data') else []

        return render_template(
            'appointments.html',
            current_user=current_user,
            appointments=appointments_list
        )
    except Exception as e:
        print(f"Appointments error: {str(e)}")
        return render_template(
            'appointments.html',
            current_user=current_user,
            appointments=[],
            error="Unable to load appointments"
        )

# app.py
# Plans page
@app.route('/plans')
@login_required
def plans():
    # Mock plans data
    plans_data = [
        {"name": "Basic Plan", "price": "500 BDT", "details": "Access to general consultation."},
        {"name": "Premium Plan", "price": "1200 BDT", "details": "Includes specialist consultation and reports."},
        {"name": "Ultimate Plan", "price": "2500 BDT", "details": "All-inclusive consultation, priority support."}
    ]

    # Fetch current user's active subscriptions from Supabase
    user_subscriptions = supabase.table("user_subscriptions")\
        .select("*")\
        .eq("user_id", str(current_user.id))\
        .eq("active", True)\
        .execute().data

    return render_template("plans.html", plans=plans_data, user_subscriptions=user_subscriptions)
# Subscribe to a plan (mock payment)
@app.route('/subscribe/<plan_name>', methods=['POST'])
@login_required
def subscribe_plan(plan_name):
    # Check if already subscribed
    existing = supabase.table("user_subscriptions")\
        .select("*")\
        .eq("user_id", str(current_user.id))\
        .eq("plan_name", plan_name)\
        .eq("active", True)\
        .execute().data

    if existing:
        flash("You are already subscribed to this plan.", "warning")
        return redirect(url_for('plans'))

    # Save subscription
    start_date = datetime.now()
    end_date = start_date + timedelta(days=30)  # Example: 30-day subscription

    plan_points = {
        "Basic Plan": 1,
        "Premium Plan": 2,
        "Ultimate Plan": 5
    }

    supabase.table("user_subscriptions").insert({
        "user_id": str(current_user.id),
        "plan_name": plan_name,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "active": True,
        "checkup_points": plan_points.get(plan_name, 0)
    }).execute()

    flash(f"Subscribed to {plan_name} successfully!", "success")
    return redirect(url_for('plans'))

# Cancel subscription
@app.route('/cancel_subscription/<sub_id>', methods=['POST'])
@login_required
def cancel_subscription(sub_id):
    # Update subscription to inactive
    response = supabase.table("user_subscriptions")\
        .update({
            "active": False,
            "end_date": datetime.now().isoformat()
        })\
        .eq("id", sub_id)\
        .eq("user_id", str(current_user.id))\
        .execute()

    if response.data:  # If data is returned, the update succeeded
        flash("Subscription canceled successfully.", "success")
    else:
        flash("Failed to cancel subscription.", "danger")

    return redirect(url_for('plans'))

# Plan details (optional for details button)
@app.route('/plan/<plan_name>')
@login_required
def plan_details(plan_name):
    plans_data = {
        "Basic Plan": {"price": "500 BDT", "features": ["Feature A", "Feature B"]},
        "Premium Plan": {"price": "1200 BDT", "features": ["Feature C", "Feature D"]},
        "Ultimate Plan": {"price": "2500 BDT", "features": ["Feature E", "Feature F"]}
    }

    plan = plans_data.get(plan_name)
    if not plan:
        flash("Plan not found!", "danger")
        return redirect(url_for('plans'))

    return render_template("plan_details.html", plan_name=plan_name, plan=plan)


# Feature page for subscribed plans
# @app.route('/features/<plan_name>')
# @login_required
# def plan_feature(plan_name):
#     # You can also fetch subscription to ensure user is active
#     subscription = supabase.table("user_subscriptions")\
#         .select("*")\
#         .eq("user_id", current_user.id)\
#         .eq("plan_name", plan_name)\
#         .eq("active", True)\
#         .execute().data
#
#     if not subscription:
#         flash("You are not subscribed to this plan!", "danger")
#         return redirect(url_for('plans'))
#
#     # Mock features per plan
#     features_data = {
#         "Basic Plan": ["Feature A", "Feature B", "Feature C"],
#         "Premium Plan": ["Feature D", "Feature E", "Feature F"],
#         "Ultimate Plan": ["Feature G", "Feature H", "Feature I"]
#     }
#
#     features = features_data.get(plan_name, [])
#     return render_template("features.html", plan_name=plan_name, features=features)
# from flask import abort

# Feature page access based on subscription
@app.route('/features/<plan_name>')
@login_required
def features(plan_name):
    # Check subscription
    subscription = supabase.table("user_subscriptions")\
        .select("*")\
        .eq("user_id", str(current_user.id))\
        .eq("plan_name", plan_name)\
        .eq("active", True)\
        .execute().data

    if not subscription:
        flash("You need to subscribe to access this feature page.", "danger")
        return redirect(url_for('plans'))

    # Query doctors based on plan level
    if plan_name == "Basic Plan":
        doctors = supabase.table("doctors")\
            .select("*")\
            .eq("specialty", "General")\
            .execute().data
    elif plan_name == "Premium Plan":
        doctors = supabase.table("doctors")\
            .select("*")\
            .execute().data
    elif plan_name == "Ultimate Plan":
        doctors = supabase.table("doctors")\
            .select("*")\
            .execute().data
    else:
        abort(404)

    # Fetch subscription for points display
    subscription_resp = supabase.table("user_subscriptions") \
        .select("*") \
        .eq("user_id", str(current_user.id)) \
        .eq("active", True) \
        .order("start_date", desc=True) \
        .limit(1) \
        .execute()

    subscription = subscription_resp.data[0] if subscription_resp.data else None

    # Render template with doctors and subscription
    if plan_name == "Basic Plan":
        return render_template("feature_basic.html", doctors=doctors, subscription=subscription)
    elif plan_name == "Premium Plan":
        return render_template("feature_premium.html", doctors=doctors, subscription=subscription)
    elif plan_name == "Ultimate Plan":
        return render_template("feature_ultimate.html", doctors=doctors, subscription=subscription)
if __name__ == '__main__':
    app.run(debug=True)