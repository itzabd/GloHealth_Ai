import os
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps

import folium
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from folium.plugins import HeatMap, MarkerCluster
from supabase import create_client

load_dotenv()


def get_env_var(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key) or os.environ.get(key.lower()) or default


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = get_env_var("FLASK_SECRET_KEY", "dev-secret-key-change-me")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Supabase configuration
SUPABASE_URL = get_env_var("SUPABASE_URL")
SUPABASE_KEY = get_env_var("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY environment variables.")
SUPABASE_URL = SUPABASE_URL.strip().rstrip(",")
SUPABASE_KEY = SUPABASE_KEY.strip()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load ML model
model = joblib.load('results/production_model.joblib')
le = joblib.load('results/label_encoder.joblib')
feature_cols = joblib.load('results/feature_columns.joblib')

# ---- SYMPTOM LABELS ----
SYMPTOM_OVERRIDES = {
    'high_fever': 'High fever',
    'mild_fever': 'Mild fever',
    'skin_rash': 'Skin rash',
    'nodal_skin_eruptions': 'Nodal skin eruptions',
    'breathlessness': 'Shortness of breath',
    'loss_of_appetite': 'Loss of appetite',
    'abdominal_pain': 'Abdominal pain',
    'yellowish_skin': 'Yellowish skin',
    'yellowing_of_eyes': 'Yellowing of eyes',
    'receiving_blood_transfusion': 'Recent blood transfusion',
    'receiving_unsterile_injections': 'Recent unsterile injections',
    'acute_liver_failure': 'Acute liver failure',
    'swelling_of_stomach': 'Swelling of stomach',
    'swelled_lymph_nodes': 'Swelled lymph nodes',
    'malaise': 'Malaise / general discomfort',
    'blurred_and_distorted_vision': 'Blurred or distorted vision',
    'throat_irritation': 'Throat irritation',
    'redness_of_eyes': 'Redness of eyes',
    'sinus_pressure': 'Sinus pressure',
    'runny_nose': 'Runny nose',
    'congestion': 'Nasal congestion',
    'chest_pain': 'Chest pain',
    'weakness_in_limbs': 'Weakness in limbs',
    'fast_heart_rate': 'Fast heart rate',
    'pain_during_bowel_movements': 'Pain during bowel movements',
    'pain_in_anal_region': 'Pain in anal region',
    'bloody_stool': 'Bloody stool',
    'irritation_in_anus': 'Irritation in anus',
    'neck_pain': 'Neck pain',
    'dizziness': 'Dizziness',
    'cramps': 'Cramps',
    'bruising': 'Bruising',
    'obesity': 'Obesity',
    'swollen_legs': 'Swollen legs',
    'swollen_blood_vessels': 'Swollen blood vessels',
    'puffy_face_and_eyes': 'Puffy face and eyes',
    'enlarged_thyroid': 'Enlarged thyroid',
    'brittle_nails': 'Brittle nails',
    'swollen_extremeties': 'Swollen extremities',
    'excessive_hunger': 'Excessive hunger',
    'extra_marital_contacts': 'Recent extra-marital contact',
    'drying_and_tingling_lips': 'Drying and tingling lips',
    'slurred_speech': 'Slurred speech',
    'knee_pain': 'Knee pain',
    'hip_joint_pain': 'Hip joint pain',
    'muscle_weakness': 'Muscle weakness',
    'stiff_neck': 'Stiff neck',
    'swelling_joints': 'Swelling joints',
    'movement_stiffness': 'Movement stiffness',
    'spinning_movements': 'Spinning movements',
    'loss_of_balance': 'Loss of balance',
    'unsteadiness': 'Unsteadiness',
    'weakness_of_one_body_side': 'Weakness of one body side',
    'loss_of_smell': 'Loss of smell',
    'bladder_discomfort': 'Bladder discomfort',
    'continuous_feel_of_urine': 'Continuous feel of urine',
    'passage_of_gases': 'Passage of gases',
    'internal_itching': 'Internal itching',
    'depression': 'Depression',
    'irritability': 'Irritability',
    'muscle_pain': 'Muscle pain',
    'altered_sensorium': 'Altered sensorium',
    'red_spots_over_body': 'Red spots over body',
    'belly_pain': 'Belly pain',
    'abnormal_menstruation': 'Abnormal menstruation',
    'dischromic_patches': 'Discolored patches',
    'watering_from_eyes': 'Watering from eyes',
    'increased_appetite': 'Increased appetite',
    'polyuria': 'Polyuria',
    'family_history': 'Family history',
    'mucoid_sputum': 'Mucoid sputum',
    'rusty_sputum': 'Rusty sputum',
    'lack_of_concentration': 'Lack of concentration',
    'visual_disturbances': 'Visual disturbances',
    'blood_in_sputum': 'Blood in sputum',
    'prominent_veins_on_calf': 'Prominent veins on calf',
    'palpitations': 'Palpitations',
    'painful_walking': 'Painful walking',
    'pus_filled_pimples': 'Pus-filled pimples',
    'blackheads': 'Blackheads',
    'scurring': 'Scarring',
    'skin_peeling': 'Skin peeling',
    'silver_like_dusting': 'Silver-like dusting',
    'small_dents_in_nails': 'Small dents in nails',
    'inflammatory_nails': 'Inflammatory nails',
    'blister': 'Blister',
    'red_sore_around_nose': 'Red sore around nose',
    'yellow_crust_ooze': 'Yellow crust ooze',
}

def symptom_label(s):
    if s in SYMPTOM_OVERRIDES:
        return SYMPTOM_OVERRIDES[s]
    return s.replace('_', ' ').strip().capitalize()

SYMPTOM_LABELS_BN = {
    'itching': 'চুলকানি', 'skin_rash': 'ত্বকে র্যাশ', 'chills': 'শীত শীত ভাব',
    'high_fever': 'উচ্চ জ্বর', 'mild_fever': 'হালকা জ্বর', 'fatigue': 'ক্লান্তি',
    'cough': 'কাশি', 'breathlessness': 'শ্বাসকষ্ট', 'headache': 'মাথা ব্যথা',
    'dizziness': 'মাথা ঘোরা', 'vomiting': 'বমি', 'diarrhoea': 'ডায়রিয়া',
    'nausea': 'বমি বমি ভাব', 'abdominal_pain': 'পেটে ব্যথা', 'chest_pain': 'বুকে ব্যথা',
    'joint_pain': 'গাঁটে ব্যথা', 'muscle_pain': 'পেশিতে ব্যথা', 'loss_of_appetite': 'খাবারে অরুচি',
    'yellowish_skin': 'ত্বক হলদে', 'yellowing_of_eyes': 'চোখ হলদে', 'weight_loss': 'ওজন কমা',
    'sweating': 'ঘাম', 'dehydration': 'পানিশূন্যতা', 'constipation': 'কোষ্ঠকাঠিন্য',
    'back_pain': 'পিঠে ব্যথা', 'neck_pain': 'ঘাড়ে ব্যথা', 'knee_pain': 'হাঁটুতে ব্যথা',
    'hip_joint_pain': 'কোমরে ব্যথা', 'muscle_weakness': 'পেশির দুর্বলতা',
    'stiff_neck': 'শক্ত ঘাড়', 'swelling_joints': 'গাঁট ফোলা', 'movement_stiffness': 'নড়াচড়ায় কষ্ট',
    'loss_of_balance': 'ভারসাম্য হারানো', 'unsteadiness': 'অস্থিরতা',
    'loss_of_smell': 'গন্ধ না পাওয়া', 'blurred_and_distorted_vision': 'ঝাপসা দৃষ্টি',
    'redness_of_eyes': 'চোখ লাল', 'watering_from_eyes': 'চোখ দিয়ে পানি',
    'sinus_pressure': 'সাইনাসের চাপ', 'runny_nose': 'নাক দিয়ে পানি', 'congestion': 'নাক বন্ধ',
    'throat_irritation': 'গলা জ্বালা', 'phlegm': 'কফ', 'blood_in_sputum': 'কফে রক্ত',
    'fast_heart_rate': 'দ্রুত হৃদস্পন্দন', 'palpitations': 'বুক ধড়ফড়',
    'weakness_in_limbs': 'হাত-পায়ে দুর্বলতা', 'swollen_legs': 'পা ফোলা',
    'obesity': 'স্থূলতা', 'excessive_hunger': 'অতিরিক্ত ক্ষুধা', 'increased_appetite': 'বেড়ে যাওয়া ক্ষুধা',
    'polyuria': 'অতিরিক্ত প্রস্রাব', 'continuous_feel_of_urine': 'সবসময় প্রস্রাবের অনুভূতি',
    'bladder_discomfort': 'মূত্রথলিতে অস্বস্তি', 'bloody_stool': 'পায়খানায় রক্ত',
    'pain_during_bowel_movements': 'পায়খানার সময় ব্যথা', 'passage_of_gases': 'গ্যাস নির্গমন',
    'internal_itching': 'ভেতরে চুলকানি', 'irritation_in_anus': 'পায়ুপথে জ্বালা',
    'bruising': 'ক্ষতচিহ্ন', 'skin_peeling': 'ত্বক ওঠা', 'blister': 'ফোস্কা',
    'pus_filled_pimples': 'পুঁজ ভরা ব্রণ', 'blackheads': 'ব্ল্যাকহেডস', 'scurring': 'দাগ',
    'silver_like_dusting': 'রূপালী আঁশ', 'small_dents_in_nails': 'নখে ছোট গর্ত',
    'inflammatory_nails': 'নখে প্রদাহ', 'brittle_nails': 'ভাঙা নখ',
    'swollen_extremeties': 'হাত-পা ফোলা', 'puffy_face_and_eyes': 'মুখ ও চোখ ফোলা',
    'enlarged_thyroid': 'থাইরয়েড বড়', 'swollen_blood_vessels': 'রক্তনালী ফোলা',
    'prominent_veins_on_calf': 'পায়ে শিরা ফোলা', 'drying_and_tingling_lips': 'ঠোঁট শুকনো ও ঝিনঝিন',
    'slurred_speech': 'জড়ানো কথা', 'spinning_movements': 'মাথা ঘোরা',
    'weakness_of_one_body_side': 'শরীরের একপাশে দুর্বলতা', 'altered_sensorium': 'চেতনা পরিবর্তন',
    'depression': 'বিষণ্নতা', 'irritability': 'খিটখিটে ভাব', 'anxiety': 'দুশ্চিন্তা',
    'mood_swings': 'মেজাজ পরিবর্তন', 'restlessness': 'অস্থিরতা', 'lethargy': 'অলসতা',
    'red_spots_over_body': 'শরীরে লাল দাগ', 'belly_pain': 'তলপেটে ব্যথা',
    'abnormal_menstruation': 'অস্বাভাবিক মাসিক', 'dischromic_patches': 'বর্ণহীন দাগ',
    'family_history': 'পারিবারিক ইতিহাস', 'mucoid_sputum': 'শ্লেষ্মাযুক্ত কফ',
    'rusty_sputum': 'মরিচা রঙের কফ', 'lack_of_concentration': 'মনোযোগের অভাব',
    'visual_disturbances': 'দৃষ্টি সমস্যা', 'painful_walking': 'হাঁটতে ব্যথা',
    'red_sore_around_nose': 'নাকের চারপাশে লাল ঘা', 'yellow_crust_ooze': 'হলদে খোসা',
    'acute_liver_failure': 'তীব্র লিভার বিকল', 'receiving_blood_transfusion': 'রক্ত গ্রহণ',
    'receiving_unsterile_injections': 'অপরিষ্কার ইনজেকশন', 'swelling_of_stomach': 'পেট ফোলা',
    'swelled_lymph_nodes': 'লিম্ফ নোড ফোলা', 'malaise': 'শারীরিক অস্বস্তি',
    'nodal_skin_eruptions': 'গিঁটে চর্ম', 'toxic_look_(typhos)': 'টক্সিক লক্ষণ',
    'extra_marital_contacts': 'অতিরিক্ত যৌন সংস্পর্শ',
    # Remaining ~30 symptoms fall back to English via .get() default
}

SPECIALTY_BN = {
    'Cardiologist': 'হৃদরোগ বিশেষজ্ঞ', 'Neurologist': 'স্নায়ু বিশেষজ্ঞ',
    'Dermatologist': 'চর্ম বিশেষজ্ঞ', 'General Physician': 'সাধারণ চিকিৎসক',
    'Pediatrician': 'শিশু বিশেষজ্ঞ', 'Orthopedic': 'অর্থোপেডিক বিশেষজ্ঞ',
    'Cardiology': 'হৃদরোগ', 'Dermatology': 'চর্মরোগ', 'Neurology': 'স্নায়ুরোগ',
    'Internal Medicine': 'অভ্যন্তরীণ চিকিৎসা', 'Pediatrics': 'শিশু চিকিৎসা',
    'Orthopedics': 'অর্থোপেডিকস',
}

DIVISION_BN = {
    'Dhaka': 'ঢাকা', 'Chattogram': 'চট্টগ্রাম', 'Chittagong': 'চট্টগ্রাম',
    'Khulna': 'খুলনা', 'Rajshahi': 'রাজশাহী', 'Barishal': 'বরিশাল',
    'Barisal': 'বরিশাল', 'Sylhet': 'সিলেট', 'Rangpur': 'রংপুর',
    'Mymensingh': 'ময়মনসিংহ',
}

def symptom_label_bn(s):
    return SYMPTOM_LABELS_BN.get(s, symptom_label(s))

def specialty_bn(s):
    return SPECIALTY_BN.get(s, s)

def division_bn(d):
    return DIVISION_BN.get(d, d)

# ---- FLASH I18N ----
FLASH_I18N = {
    'en': {
        'flash.unauthorized': "Unauthorized access",
        'flash.err_users': "Error fetching users",
        'flash.err_doctors': "Error fetching doctors: {err}",
        'flash.doctor_added': "Doctor added successfully!",
        'flash.err_add_doctor': "Error adding doctor: {err}",
        'flash.err_doctor': "Error fetching doctor: {err}",
        'flash.doctor_updated': "Doctor updated successfully!",
        'flash.err_update_doctor': "Error updating doctor: {err}",
        'flash.doctor_deleted': "Doctor deleted successfully!",
        'flash.err_delete_doctor': "Error deleting doctor: {err}",
        'flash.err_appts': "Error fetching appointments",
        'flash.appt_added': "Appointment added successfully!",
        'flash.err_add_appt': "Error adding appointment: {err}",
        'flash.err_appt': "Error fetching appointment: {err}",
        'flash.appt_updated': "Appointment updated successfully!",
        'flash.err_update_appt': "Error updating appointment: {err}",
        'flash.appt_deleted': "Appointment deleted successfully!",
        'flash.err_delete_appt': "Error deleting appointment: {err}",
        'flash.user_updated': "User updated successfully!",
        'flash.user_deleted': "User deleted successfully!",
        'flash.err_delete_user': "Error deleting user: {err}",
        'flash.settings_updated': "Settings updated successfully",
        'flash.err_settings': "Error updating settings",
        'flash.email_pwd_required': "Email and password are required.",
        'flash.invalid_credentials': "Invalid email or password. Please try again.",
        'flash.signup_required': "Full name, email, and password are required.",
        'flash.pwd_min_len': "Password must be at least 6 characters long.",
        'flash.create_acc_failed': "Failed to create account. Please try again.",
        'flash.acc_created_welcome': "Account created successfully! Welcome to GloHealth AI.",
        'flash.acc_created_login': "Account created successfully! Please log in.",
        'flash.acc_exists': "An account with this email already exists. Please log in.",
        'flash.signup_err': "Signup error: {err}",
        'flash.doctor_not_found': "Doctor not found.",
        'flash.select_date_time': "Please select a date and time.",
        'flash.appt_booked': "Appointment booked successfully!",
        'flash.subscribed': "Subscribed to {plan} successfully!",
        'flash.sub_cancelled': "Subscription canceled successfully.",
        'flash.sub_cancel_failed': "Failed to cancel subscription.",
        'flash.plan_not_found': "Plan not found!",
        'flash.feature_sub_required': "You need to subscribe to access this feature page.",
    },
    'bn': {
        'flash.unauthorized': "অননুমোদিত প্রবেশ",
        'flash.err_users': "ব্যবহারকারী আনতে ত্রুটি",
        'flash.err_doctors': "ডাক্তার আনতে ত্রুটি: {err}",
        'flash.doctor_added': "ডাক্তার সফলভাবে যোগ করা হয়েছে!",
        'flash.err_add_doctor': "ডাক্তার যোগ করতে ত্রুটি: {err}",
        'flash.err_doctor': "ডাক্তার আনতে ত্রুটি: {err}",
        'flash.doctor_updated': "ডাক্তার সফলভাবে আপডেট করা হয়েছে!",
        'flash.err_update_doctor': "ডাক্তার আপডেট করতে ত্রুটি: {err}",
        'flash.doctor_deleted': "ডাক্তার সফলভাবে মুছে ফেলা হয়েছে!",
        'flash.err_delete_doctor': "ডাক্তার মুছতে ত্রুটি: {err}",
        'flash.err_appts': "অ্যাপয়েন্টমেন্ট আনতে ত্রুটি",
        'flash.appt_added': "অ্যাপয়েন্টমেন্ট সফলভাবে যোগ করা হয়েছে!",
        'flash.err_add_appt': "অ্যাপয়েন্টমেন্ট যোগ করতে ত্রুটি: {err}",
        'flash.err_appt': "অ্যাপয়েন্টমেন্ট আনতে ত্রুটি: {err}",
        'flash.appt_updated': "অ্যাপয়েন্টমেন্ট সফলভাবে আপডেট করা হয়েছে!",
        'flash.err_update_appt': "অ্যাপয়েন্টমেন্ট আপডেট করতে ত্রুটি: {err}",
        'flash.appt_deleted': "অ্যাপয়েন্টমেন্ট সফলভাবে মুছে ফেলা হয়েছে!",
        'flash.err_delete_appt': "অ্যাপয়েন্টমেন্ট মুছতে ত্রুটি: {err}",
        'flash.user_updated': "ব্যবহারকারী সফলভাবে আপডেট করা হয়েছে!",
        'flash.user_deleted': "ব্যবহারকারী সফলভাবে মুছে ফেলা হয়েছে!",
        'flash.err_delete_user': "ব্যবহারকারী মুছতে ত্রুটি: {err}",
        'flash.settings_updated': "সেটিংস সফলভাবে আপডেট করা হয়েছে",
        'flash.err_settings': "সেটিংস আপডেট করতে ত্রুটি",
        'flash.email_pwd_required': "ইমেইল এবং পাসওয়ার্ড আবশ্যক।",
        'flash.invalid_credentials': "অবৈধ ইমেইল বা পাসওয়ার্ড। আবার চেষ্টা করুন।",
        'flash.signup_required': "পুরো নাম, ইমেইল এবং পাসওয়ার্ড আবশ্যক।",
        'flash.pwd_min_len': "পাসওয়ার্ড কমপক্ষে ৬ অক্ষরের হতে হবে।",
        'flash.create_acc_failed': "অ্যাকাউন্ট তৈরি করতে ব্যর্থ হয়েছে। আবার চেষ্টা করুন।",
        'flash.acc_created_welcome': "অ্যাকাউন্ট সফলভাবে তৈরি হয়েছে! GloHealth AI-তে স্বাগতম।",
        'flash.acc_created_login': "অ্যাকাউন্ট সফলভাবে তৈরি হয়েছে! অনুগ্রহ করে লগ ইন করুন।",
        'flash.acc_exists': "এই ইমেইল দিয়ে ইতিমধ্যে একটি অ্যাকাউন্ট বিদ্যমান। অনুগ্রহ করে লগ ইন করুন।",
        'flash.signup_err': "নিবন্ধন ত্রুটি: {err}",
        'flash.doctor_not_found': "ডাক্তার পাওয়া যায়নি।",
        'flash.select_date_time': "অনুগ্রহ করে একটি তারিখ ও সময় নির্বাচন করুন।",
        'flash.appt_booked': "অ্যাপয়েন্টমেন্ট সফলভাবে বুক করা হয়েছে!",
        'flash.subscribed': "{plan}-এ সফলভাবে সাবস্ক্রাইব করা হয়েছে!",
        'flash.sub_cancelled': "সাবস্ক্রিপশন সফলভাবে বাতিল করা হয়েছে।",
        'flash.sub_cancel_failed': "সাবস্ক্রিপশন বাতিল করতে ব্যর্থ হয়েছে।",
        'flash.plan_not_found': "প্ল্যান পাওয়া যায়নি!",
        'flash.feature_sub_required': "এই ফিচার পেজটি ব্যবহার করতে আপনাকে সাবস্ক্রাইব করতে হবে।",
    }
}


def t_flash(key, **kw):
    lang = session.get('lang', 'en')
    s = FLASH_I18N.get(lang, {}).get(key) or FLASH_I18N['en'].get(key, key)
    return s.format(**kw) if kw else s


@app.before_request
def _set_lang():
    if 'lang' not in session:
        session['lang'] = request.cookies.get('lang', 'en')



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
        if not user_response or not user_response.user:
            return None

        # Get profile data
        profile_resp = supabase.from_('user_profiles') \
            .select('*') \
            .eq('id', user_id) \
            .maybe_single() \
            .execute()
        profile = (profile_resp.data if hasattr(profile_resp, 'data') and profile_resp.data else {}) or {}

        user_metadata = user_response.user.user_metadata or {}
        name = user_metadata.get('full_name') or user_metadata.get('name') or (profile.get('full_name', '') if isinstance(profile, dict) else '')

        # Return User instance with is_admin
        return User(
            id=user_id,
            email=user_response.user.email,
            name=name,
            address=profile,
            is_admin=profile.get('is_admin', False) if isinstance(profile, dict) else False
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
    return render_template('admin_dashboard.html', users=users, appointments=appointments, hide_nav=True)


@app.route('/admin/users')
@login_required
def admin_users():
    if not getattr(current_user, 'is_admin', False):
        flash(t_flash("flash.unauthorized"), "danger")
        return redirect(url_for('dashboard'))

    try:
        users_resp = supabase.from_("user_profiles").select("*").execute()
        users = users_resp.data if hasattr(users_resp, "data") else []
        return render_template("admin_users.html", users=users)
    except Exception as e:
        print(f"Admin Users error: {e}")
        flash(t_flash("flash.err_users"), "danger")
        return redirect(url_for("admin_dashboard"))

# View all doctors
@app.route("/admin/doctors")
@login_required
def admin_doctors():
    if not getattr(current_user, "is_admin", False):
        flash(t_flash("flash.unauthorized"), "danger")
        return redirect(url_for("dashboard"))

    try:
        resp = supabase.from_("doctors").select("*").execute()
        doctors = resp.data if hasattr(resp, "data") else []
        return render_template("admin_doctors.html", doctors=doctors)
    except Exception as e:
        flash(t_flash("flash.err_doctors", err=str(e)), "danger")
        return redirect(url_for("admin_dashboard"))


# Add doctor page
@app.route("/admin/doctors/add", methods=["GET", "POST"])
@login_required
def add_doctor_page():
    if not getattr(current_user, "is_admin", False):
        flash(t_flash("flash.unauthorized"), "danger")
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
            flash(t_flash("flash.doctor_added"), "success")
            return redirect(url_for("admin_doctors"))
        except Exception as e:
            flash(t_flash("flash.err_add_doctor", err=str(e)), "danger")

    return render_template("add_doctor.html")


# Edit doctor page
@app.route("/admin/doctors/edit/<doctor_id>", methods=["GET", "POST"])
@login_required
def edit_doctor_page(doctor_id):
    if not getattr(current_user, "is_admin", False):
        flash(t_flash("flash.unauthorized"), "danger")
        return redirect(url_for("admin_doctors"))

    try:
        resp = supabase.from_("doctors").select("*").eq("id", doctor_id).single().execute()
        doctor = resp.data
    except Exception as e:
        flash(t_flash("flash.err_doctor", err=str(e)), "danger")
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
            flash(t_flash("flash.doctor_updated"), "success")
            return redirect(url_for("admin_doctors"))
        except Exception as e:
            flash(t_flash("flash.err_update_doctor", err=str(e)), "danger")

    return render_template("edit_doctor.html", doctor=doctor)


# Delete doctor page
@app.route("/admin/doctors/delete/<doctor_id>", methods=["POST"])
@login_required
def delete_doctor_page(doctor_id):
    if not getattr(current_user, "is_admin", False):
        flash(t_flash("flash.unauthorized"), "danger")
        return redirect(url_for("admin_doctors"))

    try:
        supabase.from_("doctors").delete().eq("id", doctor_id).execute()
        flash(t_flash("flash.doctor_deleted"), "success")
    except Exception as e:
        flash(t_flash("flash.err_delete_doctor", err=str(e)), "danger")

    return redirect(url_for("admin_doctors"))
# View all appointments
@app.route('/admin/appointments')
@login_required
def admin_appointments():
    if not getattr(current_user, 'is_admin', False):
        flash(t_flash("flash.unauthorized"), "danger")
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
        flash(t_flash("flash.err_appts"), "danger")
        return redirect(url_for('admin_dashboard'))


# Add appointment
@app.route("/admin/appointments/add", methods=["GET", "POST"])
@login_required
def add_appointment_page():
    if not getattr(current_user, "is_admin", False):
        flash(t_flash("flash.unauthorized"), "danger")
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

            flash(t_flash("flash.appt_added"), "success")
            return redirect(url_for("admin_appointments"))
        except Exception as e:
            flash(t_flash("flash.err_add_appt", err=str(e)), "danger")

    return render_template("add_appointment.html", doctors=doctors, users=users)

# Edit appointment
@app.route("/admin/appointments/edit/<appointment_id>", methods=["GET", "POST"])
@login_required
def edit_appointment_page(appointment_id):
    if not getattr(current_user, "is_admin", False):
        flash(t_flash("flash.unauthorized"), "danger")
        return redirect(url_for("admin_appointments"))

    try:
        resp = supabase.from_("appointments").select("*").eq("id", appointment_id).single().execute()
        appointment = resp.data

        doctors_resp = supabase.from_("doctors").select("*").execute()
        doctors = doctors_resp.data if hasattr(doctors_resp, "data") else []

        users_resp = supabase.from_("user_profiles").select("*").execute()
        users = users_resp.data if hasattr(users_resp, "data") else []

    except Exception as e:
        flash(t_flash("flash.err_appt", err=str(e)), "danger")
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

            flash(t_flash("flash.appt_updated"), "success")
            return redirect(url_for("admin_appointments"))
        except Exception as e:
            flash(t_flash("flash.err_update_appt", err=str(e)), "danger")

    return render_template("edit_appointment.html", appointment=appointment, doctors=doctors, users=users)


# Delete appointment
@app.route("/admin/appointments/delete/<appointment_id>", methods=["POST"])
@login_required
def delete_appointment_page(appointment_id):
    if not getattr(current_user, "is_admin", False):
        flash(t_flash("flash.unauthorized"), "danger")
        return redirect(url_for("admin_appointments"))

    try:
        supabase.from_("appointments").delete().eq("id", appointment_id).execute()
        flash(t_flash("flash.appt_deleted"), "success")
    except Exception as e:
        flash(t_flash("flash.err_delete_appt", err=str(e)), "danger")

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

        flash(t_flash("flash.user_updated"), "success")
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

        flash(t_flash("flash.user_deleted"), "success")
    except Exception as e:
        flash(t_flash("flash.err_delete_user", err=str(e)), "danger")

    return redirect(url_for("admin_users"))

# Add Doctor

@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
def admin_settings():
    if not getattr(current_user, 'is_admin', False):
        flash(t_flash("flash.unauthorized"), "danger")
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

            flash(t_flash("flash.settings_updated"), "success")
            return redirect(url_for('admin_settings'))

        return render_template('admin_settings.html', settings=settings)
    except Exception as e:
        print(f"Update settings error: {str(e)}")
        flash(t_flash("flash.err_settings"), "danger")
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
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html', current_user=current_user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '').strip()

            if not email or not password:
                flash(t_flash("flash.email_pwd_required"), "danger")
                return render_template('auth/login.html')

            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if response and response.session:
                session.update({
                    'supabase_access_token': response.session.access_token,
                    'supabase_refresh_token': response.session.refresh_token
                })

                user = load_user(response.user.id)
                if user:
                    login_user(user)
                    return redirect(url_for('dashboard'))

            flash(t_flash("flash.invalid_credentials"), "danger")

        except Exception as e:
            print(f"Login error: {str(e)}")
            flash(t_flash("flash.invalid_credentials"), "danger")
            return render_template('auth/login.html')

    return render_template('auth/login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '').strip()
            address_line1 = request.form.get('address_line1', '').strip()
            address_line2 = request.form.get('address_line2', '').strip()
            city = request.form.get('city', '').strip()
            division = request.form.get('division', '').strip()
            postal_code = request.form.get('postal_code', '').strip()

            if not name or not email or not password:
                flash(t_flash("flash.signup_required"), "danger")
                return render_template('auth/signup.html')

            if len(password) < 6:
                flash(t_flash("flash.pwd_min_len"), "danger")
                return render_template('auth/signup.html')

            auth_response = supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "full_name": name,
                        "division": division
                    }
                }
            })

            if not auth_response or not auth_response.user:
                flash(t_flash("flash.create_acc_failed"), "danger")
                return render_template('auth/signup.html')

            user = auth_response.user

            # Insert profile data
            profile_data = {
                "id": user.id,
                "address_line1": address_line1,
                "address_line2": address_line2 or None,
                "city": city,
                "division": division,
                "postal_code": postal_code,
                "full_name": name,
                "email": user.email
            }
            supabase.from_("user_profiles").insert(profile_data).execute()

            # Handle session / auto-login
            session_obj = auth_response.session
            if not session_obj:
                try:
                    sign_in_resp = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password
                    })
                    if sign_in_resp:
                        session_obj = sign_in_resp.session
                except Exception as sign_in_err:
                    print(f"Auto-login notice after signup: {sign_in_err}")

            if session_obj:
                session.update({
                    'supabase_access_token': session_obj.access_token,
                    'supabase_refresh_token': session_obj.refresh_token
                })

                app_user = load_user(user.id)
                if app_user:
                    login_user(app_user)
                    flash(t_flash("flash.acc_created_welcome"), "success")
                    return redirect(url_for('dashboard'))

            flash(t_flash("flash.acc_created_login"), "success")
            return redirect(url_for('login'))

        except Exception as e:
            error_msg = str(e)
            print(f"Signup error: {error_msg}")
            if "User already registered" in error_msg or "user_already_exists" in error_msg:
                flash(t_flash("flash.acc_exists"), "warning")
                return redirect(url_for('login'))
            elif "Password should be at least 6 characters" in error_msg or "weak_password" in error_msg:
                flash(t_flash("flash.pwd_min_len"), "danger")
            else:
                flash(t_flash("flash.signup_err", err=error_msg), "danger")
            return render_template('auth/signup.html')

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

        from datetime import datetime, timezone

        total_predictions = len(predictions)
        predictions_delta = ""
        if len(predictions) >= 2:
            try:
                one_week = 7 * 24 * 3600
                now = datetime.now(timezone.utc).timestamp()
                recent = sum(1 for p in predictions
                             if p.get('timestamp') and
                             now - datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')).timestamp() < one_week)
                predictions_delta = f"+{recent} this week" if recent else ""
            except Exception:
                predictions_delta = ""

        upcoming = [a for a in appointments if a.get('status') in ('confirmed', 'pending')]
        next_appointment = upcoming[0] if upcoming else None

        plan_resp = supabase.table("user_subscriptions") \
            .select("plan_name") \
            .eq("user_id", str(current_user.id)) \
            .eq("active", True) \
            .order("start_date", desc=True) \
            .limit(1) \
            .execute()
        plan_name = plan_resp.data[0]['plan_name'] if plan_resp.data else "Free"
        health_score = 82 if total_predictions else 0

        kpis = {
            "total_predictions": total_predictions,
            "predictions_delta": predictions_delta,
            "upcoming_count": len(upcoming),
            "next_appointment_label": (next_appointment.get('scheduled_time', '')[:16]
                                       if next_appointment else "None scheduled"),
            "plan_name": plan_name,
            "plan_status": "active" if plan_name != "Free" else "inactive",
            "health_score": health_score,
        }

        health_tips = [
            {"icon": "water_drop", "text": "Drink 2-3L water daily"},
            {"icon": "directions_walk", "text": "30 min walk after meals"},
            {"icon": "bedtime", "text": "Sleep 7-8 hours"},
        ]

        return render_template('dashboard.html',
                               current_user=current_user,
                               predictions=predictions,
                               appointments=appointments,
                               kpis=kpis,
                               next_appointment=next_appointment,
                               health_tips=health_tips)
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
                symptom_groups[category].append({
                    "key": symptom,
                    "label": symptom_label(symptom),
                    "label_bn": symptom_label_bn(symptom),
                })
                matched = True
                break
        if not matched:
            symptom_groups['Other'].append({
                "key": symptom,
                "label": symptom_label(symptom),
                "label_bn": symptom_label_bn(symptom),
            })

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
    DIVISION_COORDS = {
        'Dhaka': [23.8103, 90.4125],
        'Chattogram': [22.3569, 91.7832],
        'Chittagong': [22.3569, 91.7832],
        'Khulna': [22.8456, 89.5403],
        'Rajshahi': [24.3745, 88.6042],
        'Barishal': [22.7010, 90.3535],
        'Barisal': [22.7010, 90.3535],
        'Sylhet': [24.8910, 91.8710],
        'Rangpur': [25.7439, 89.2752],
        'Mymensingh': [24.7471, 90.4203]
    }

    division_counts = {}
    division_diseases = {}
    disease_counts = {}

    try:
        response = supabase.from_('location_insights').select('*').execute()
        rows = response.data if (response and hasattr(response, 'data') and response.data) else []
        for r in rows:
            div = r.get('division')
            if not div:
                continue
            div = 'Chattogram' if div == 'Chittagong' else ('Barishal' if div == 'Barisal' else div)
            cnt = r.get('case_count', 1) or 1
            dis = r.get('disease', 'Unspecified')

            division_counts[div] = division_counts.get(div, 0) + cnt
            if div not in division_diseases:
                division_diseases[div] = {}
            division_diseases[div][dis] = division_diseases[div].get(dis, 0) + cnt
            disease_counts[dis] = disease_counts.get(dis, 0) + cnt
    except Exception as e:
        print(f"Location insights query fallback: {e}")

    # Fallback to sample data matching spec if empty
    if not division_counts:
        division_counts = {
            'Dhaka': 5240, 'Chattogram': 3180, 'Khulna': 1420, 'Rajshahi': 1050,
            'Barishal': 780, 'Sylhet': 620, 'Rangpur': 480, 'Mymensingh': 410
        }
        fallback_top_diseases = {
            'Dhaka': 'Dengue Fever', 'Chattogram': 'Malaria', 'Khulna': 'Cholera',
            'Rajshahi': 'Typhoid', 'Barishal': 'Diarrhea', 'Sylhet': 'Influenza',
            'Rangpur': 'Pneumonia', 'Mymensingh': 'Hepatitis'
        }
        division_diseases = {k: {v: division_counts[k]} for k, v in fallback_top_diseases.items()}
        disease_counts = {
            'Dengue': 4120, 'Malaria': 2340, 'Typhoid': 1890, 'Influenza': 1420, 'Diarrhea': 980
        }

    total_cases = sum(division_counts.values()) or 1

    # Build division_data list of dicts: {name, lat, lng, cases, top_disease, pct_of_total}
    division_data = []
    sorted_divisions = sorted(division_counts.items(), key=lambda x: x[1], reverse=True)
    for div_name, count in sorted_divisions:
        coords = DIVISION_COORDS.get(div_name, [23.6850, 90.3563])
        div_dis_map = division_diseases.get(div_name, {})
        top_d = max(div_dis_map.items(), key=lambda x: x[1])[0] if div_dis_map else 'Dengue'
        pct = round((count / total_cases * 100), 1)
        division_data.append({
            'name': div_name,
            'lat': coords[0],
            'lng': coords[1],
            'cases': count,
            'top_disease': top_d,
            'pct_of_total': pct
        })

    # Build top_diseases list: [{name, count}, ...] top 5
    if disease_counts:
        sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_diseases = [{'name': d, 'count': c} for d, c in sorted_diseases]
    else:
        top_diseases = [
            {'name': 'Dengue', 'count': 4120},
            {'name': 'Malaria', 'count': 2340},
            {'name': 'Typhoid', 'count': 1890},
            {'name': 'Influenza', 'count': 1420},
            {'name': 'Diarrhea', 'count': 980}
        ]

    # Build kpis dict: {total_cases, top_division, top_division_pct, top_disease, top_disease_count, active_outbreaks}
    top_div = division_data[0] if division_data else {'name': 'Dhaka', 'pct_of_total': 42}
    top_dis = top_diseases[0] if top_diseases else {'name': 'Dengue', 'count': 3120}

    kpis = {
        'total_cases': total_cases,
        'top_division': top_div['name'],
        'top_division_pct': top_div['pct_of_total'],
        'top_disease': top_dis['name'],
        'top_disease_count': top_dis['count'],
        'active_outbreaks': 3
    }

    return render_template(
        'geo_insights.html',
        current_user=current_user,
        kpis=kpis,
        division_data=division_data,
        top_diseases=top_diseases
    )

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

    for d in doctors_list:
        d['specialty_bn'] = specialty_bn(d.get('specialty', ''))
        d['division_bn'] = division_bn(d.get('division', ''))

    return render_template('doctors.html', doctors=doctors_list)
# Book appointment route
@app.route('/book_appointment/<doctor_id>', methods=['GET', 'POST'])
@login_required
def book_appointment(doctor_id):
    # Fetch doctor from Supabase
    doc_resp = supabase.table('doctors').select('*').eq('id', doctor_id).execute()
    doctor = doc_resp.data[0] if doc_resp.data else None
    if not doctor:
        flash(t_flash('flash.doctor_not_found'), 'danger')
        return redirect(url_for('doctors'))

    if request.method == 'POST':
        date_str = request.form.get('date')       # e.g. "2026-09-23"
        time_str = request.form.get('time')       # e.g. "10:00"
        reason = request.form.get('reason', '').strip()
        consult_type = request.form.get('consult_type', 'video')

        if not date_str or not time_str:
            flash(t_flash('flash.select_date_time'), 'warning')
            return redirect(url_for('book_appointment', doctor_id=doctor_id))

        scheduled_time = f"{date_str}T{time_str}:00"

        supabase.table('appointments').insert({
            'user_id': str(current_user.id),
            'doctor_id': doctor_id,
            'scheduled_time': scheduled_time,
            'status': 'pending',
            'payment_status': 'unpaid',
            'user_name': getattr(current_user, 'name', '') or '',
            'user_email': getattr(current_user, 'email', '') or '',
            'doctor_name': doctor.get('name', ''),
        }).execute()

        flash(t_flash('flash.appt_booked'), 'success')
        return redirect(url_for('appointments'))

    # Build next 7 days for date chips
    from datetime import date as date_cls, timedelta as td
    today = date_cls.today()
    days = []
    for i in range(7):
        d = today + td(days=i)
        days.append({
            'iso': d.isoformat(),
            'day_short': d.strftime('%a'),
            'day_num': d.day,
            'is_today': i == 0,
        })

    # Static time slots (with some marked disabled as mock)
    slots = ['09:00','09:30','10:00','10:30','11:00','11:30','12:00','12:30',
             '14:00','14:30','15:00','15:30']
    disabled_slots = ['09:30', '11:00', '15:00']

    return render_template('book_appointment.html',
                           doctor=doctor,
                           days=days,
                           slots=slots,
                           disabled_slots=disabled_slots)

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
    plans_data = [
        {"name": "Free", "backend_name": "Free", "price": "0",
         "details": "Essential diagnostic screening"},
        {"name": "Plus", "backend_name": "Premium Plan", "price": "499",
         "details": "Comprehensive proactive health care"},
        {"name": "Pro", "backend_name": "Ultimate Plan", "price": "1299",
         "details": "For families & continuous monitoring"},
    ]

    # Get the single active subscription (if any)
    rows = supabase.table("user_subscriptions") \
        .select("*") \
        .eq("user_id", str(current_user.id)) \
        .eq("active", True) \
        .order("start_date", desc=True) \
        .limit(1) \
        .execute().data

    current_plan_name = rows[0]["plan_name"] if rows else "Free"
    current_sub_id = rows[0]["id"] if rows else None
    end_date = rows[0]["end_date"][:10] if rows and rows[0].get("end_date") else None

    return render_template("plans.html",
                           plans=plans_data,
                           current_plan_name=current_plan_name,
                           current_sub_id=current_sub_id,
                           end_date=end_date)


# Subscribe to a plan (mock payment)
@app.route('/subscribe/<plan_name>', methods=['POST'])
@login_required
def subscribe_plan(plan_name):
    # Deactivate ALL existing active subscriptions for this user first
    supabase.table("user_subscriptions") \
        .update({"active": False, "end_date": datetime.now().isoformat()}) \
        .eq("user_id", str(current_user.id)) \
        .eq("active", True) \
        .execute()

    # Insert the new active subscription
    start_date = datetime.now()
    end_date = start_date + timedelta(days=30)
    plan_points = {"Basic Plan": 1, "Premium Plan": 2, "Ultimate Plan": 5}

    supabase.table("user_subscriptions").insert({
        "user_id": str(current_user.id),
        "plan_name": plan_name,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "active": True,
        "checkup_points": plan_points.get(plan_name, 0)
    }).execute()

    flash(t_flash('flash.subscribed', plan=plan_name), "success")
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
        flash(t_flash("flash.sub_cancelled"), "success")
    else:
        flash(t_flash("flash.sub_cancel_failed"), "danger")

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
        flash(t_flash("flash.plan_not_found"), "danger")
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
        flash(t_flash("flash.feature_sub_required"), "danger")
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

@app.route('/set_language/<lang>', methods=['POST', 'GET'])
def set_language(lang):
    if lang in ('en', 'bn'):
        session['lang'] = lang
        resp = make_response('', 204)
        resp.set_cookie('lang', lang, max_age=31536000, samesite='Lax')
        return resp
    return ('', 400)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)