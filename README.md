# GloHealth AI - AI-Powered Health Symptom Reporting System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-3.1.1-blue.svg)](https://flask.palletsprojects.com/)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

**GloHealth AI** is a comprehensive, full-stack healthcare application that leverages machine learning to predict diseases based on patient-reported symptoms. The system is designed to:

- **Empower individuals** with quick, AI-driven health insights for early disease detection
- **Support public health authorities** with geospatial and temporal disease trend analysis
- **Connect patients with healthcare providers** through an integrated appointment booking system
- **Provide transparency** via explainable AI using feature importance visualizations

The application combines predictive accuracy with geographical intelligence to identify emerging health patterns at the regional level, enabling data-driven resource allocation and public health response strategies.

---

## Key Features

### 👤 User-Facing Features

| Feature | Description |
|---------|-------------|
| **Symptom Prediction** | Input symptoms to receive instant disease predictions with confidence scores |
| **Feature Transparency** | View visual explanations of which symptoms influence predictions |
| **Geospatial Analysis** | Explore seasonal and regional disease trends with interactive maps |
| **Appointment Booking** | Search, filter, and book appointments with specialized doctors |
| **Subscription Plans** | Choose from Basic, Premium, and Ultimate plans with tiered benefits |
| **Prediction History** | Track past predictions and consultation records |

### 👨‍💼 Admin Dashboard Features

| Feature | Description |
|---------|-------------|
| **User Management** | View, edit, and manage user profiles and permissions |
| **Doctor Management** | Add, edit, and manage doctor profiles with specialties and availability |
| **Appointment Control** | Schedule, modify, and cancel appointments with real-time updates |
| **System Analytics** | Monitor usage metrics and disease prevalence trends |
| **Settings Configuration** | Manage system-wide settings and support parameters |

### 🤖 AI & Analytics Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Evaluation** | Random Forest, XGBoost, SVM, Logistic Regression, Gradient Boosting |
| **Class Imbalance Handling** | SMOTE + Undersampling for robust predictions on imbalanced datasets |
| **Location-Based Boosting** | Adjust predictions using regional disease prevalence data |
| **Seasonal Adjustments** | Account for temporal patterns in disease occurrence |
| **Feature Importance Analysis** | Identify most influential symptoms for each disease |

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│        (HTML/CSS/Bootstrap - Responsive UI)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Flask Web Application                          │
│  Routes: /prediction, /predict, /doctors, /appointments    │
│  Authentication: Flask-Login + Supabase Auth               │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│          Business Logic & ML Pipeline                       │
│  ├─ Disease Prediction Engine                              │
│  ├─ Location-based Boosting                               │
│  ├─ Geospatial Analysis                                   │
│  └─ Feature Importance Extraction                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│         Data Layer - Supabase (PostgreSQL)                 │
│  ├─ User Profiles & Authentication                        │
│  ├─ Predictions & Results                                 │
│  ├─ Doctors & Appointments                                │
│  ├─ Location Insights                                     │
│  ├─ User Subscriptions                                    │
│  └─ System Settings                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend & ML
- **Framework:** Python 3.8+, Flask 3.1.1
- **Machine Learning:** scikit-learn, XGBoost, imbalanced-learn
- **Data Processing:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn, Folium

### Frontend
- **Markup:** HTML5
- **Styling:** CSS3, Bootstrap
- **Client-Side:** JavaScript (form submission, geolocation)

### Database & Authentication
- **Database:** Supabase (PostgreSQL)
- **Authentication:** Supabase Auth (Email/Password)
- **Session Management:** Flask-Login

### Deployment & DevOps
- **Server:** Gunicorn
- **Hosting:** Render / Local Server
- **Python Runtime:** 3.11.x (specified in `runtime.txt`)

### Dependencies
See [requirements.txt](requirements.txt) for complete list (67 packages)

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Pip (Python package manager)
- PostgreSQL database (via Supabase)

### Step 1: Clone the Repository
```bash
git clone https://github.com/itzabd/GloHealth_Ai.git
cd GloHealth_Ai
```

### Step 2: Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration
Create a `.env` file in the project root:
```env
# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development
FLASK_APP=app.py

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key

# Database
DATABASE_URL=your_database_url

# Optional: Email Configuration
MAIL_SERVER=your_smtp_server
MAIL_PORT=587
MAIL_USERNAME=your_email
MAIL_PASSWORD=your_password
```

### Step 5: Initialize Database
```bash
# Run Supabase migrations
python supabase_setup.py

# Verify installation
python test_install.py
```

### Step 6: Train ML Models (Optional)
```bash
# Place training data in data/ directory
python train_model.py
```

### Step 7: Run Application
```bash
# Development Server
python app.py

# Production Server (with Gunicorn)
gunicorn app:app --workers 4 --bind 0.0.0.0:8000
```

Access the application at: `http://localhost:5000`

---

## Configuration

### Database Setup

**Required Supabase Tables:**

```sql
-- User Profiles
CREATE TABLE user_profiles (
  id UUID PRIMARY KEY,
  full_name VARCHAR,
  email VARCHAR,
  city VARCHAR,
  division VARCHAR,
  postal_code VARCHAR,
  is_admin BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Predictions
CREATE TABLE predictions (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID REFERENCES user_profiles(id),
  symptoms JSONB,
  top_prediction VARCHAR,
  confidence FLOAT,
  full_results JSONB,
  division VARCHAR,
  latitude FLOAT,
  longitude FLOAT,
  zip_code VARCHAR,
  timestamp TIMESTAMP DEFAULT NOW()
);

-- Location Insights
CREATE TABLE location_insights (
  id BIGSERIAL PRIMARY KEY,
  division VARCHAR,
  disease VARCHAR,
  confidence_score FLOAT,
  prevalence_score FLOAT DEFAULT 0.001,
  case_count INTEGER DEFAULT 1,
  latitude FLOAT,
  longitude FLOAT,
  zip_code VARCHAR,
  last_updated TIMESTAMP DEFAULT NOW()
);

-- Doctors
CREATE TABLE doctors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR,
  specialty VARCHAR,
  division VARCHAR,
  district VARCHAR,
  hospital VARCHAR,
  consultation_fee DECIMAL,
  availability TEXT,
  contact VARCHAR,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Appointments
CREATE TABLE appointments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES user_profiles(id),
  doctor_id UUID REFERENCES doctors(id),
  scheduled_time TIMESTAMP,
  status VARCHAR DEFAULT 'pending',
  payment_status VARCHAR DEFAULT 'unpaid',
  created_at TIMESTAMP DEFAULT NOW()
);

-- User Subscriptions
CREATE TABLE user_subscriptions (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID REFERENCES user_profiles(id),
  plan_name VARCHAR,
  start_date TIMESTAMP,
  end_date TIMESTAMP,
  active BOOLEAN DEFAULT TRUE,
  checkup_points INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

-- System Settings
CREATE TABLE system_settings (
  id BIGSERIAL PRIMARY KEY,
  site_name VARCHAR,
  support_email VARCHAR,
  checkup_fee DECIMAL,
  updated_at TIMESTAMP DEFAULT NOW()
);
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `FLASK_SECRET_KEY` | Flask session encryption key | ✅ |
| `SUPABASE_URL` | Supabase project URL | ✅ |
| `SUPABASE_KEY` | Supabase anonymous API key | ✅ |
| `FLASK_ENV` | development/production | ❌ |

---

## Usage

### 1. User Registration & Login
1. Navigate to `/signup`
2. Provide email, password, and division information
3. Account is created in Supabase Auth
4. Profile data stored in `user_profiles` table

### 2. Disease Prediction
1. Go to `/prediction`
2. Select symptoms from categorized lists
3. Provide location (division + coordinates)
4. Submit for AI prediction
5. View top 3 predictions with confidence scores

### 3. Booking Appointments
1. Browse available doctors at `/doctors`
2. Filter by division and specialty
3. Click "Book Appointment" on doctor profile
4. Select subscription plan if needed
5. Choose appointment date/time
6. Confirm payment or use free checkup points

### 4. Admin Functions
1. Login with admin account
2. Access `/admin/dashboard`
3. Manage users, doctors, appointments, and settings

---

## Project Structure

```
GloHealth_Ai/
├── app.py                          # Main Flask application
├── train_model.py                  # ML model training
├── data_prep.py                    # Data preprocessing
├── geo_analysis.py                 # Geospatial analysis
├── supabase_setup.py               # Database init
├── requirements.txt                # Dependencies
├── runtime.txt                     # Python version
├── static/                         # CSS, JS, images
├── templates/                      # HTML templates
├── data/                           # Training datasets
├── results/                        # Model outputs
└── .env                            # Environment variables
```

---

## Machine Learning Models

### Training Pipeline

The system evaluates **6 different machine learning algorithms**:

| Model | Best For |
|-------|----------|
| Random Forest | High accuracy, feature importance |
| XGBoost | Fast training, gradient boosting |
| SVM | Non-linear patterns |
| Logistic Regression | Interpretability |
| Gradient Boosting | Sequential learning |
| Extra Trees | Faster training |

### Model Selection
Best model selected based on **test F1-score (weighted)**

### Output Artifacts
- `production_model.joblib` - Best model for deployment
- `label_encoder.joblib` - Disease class encoder
- `feature_columns.joblib` - Feature list
- Confusion matrices and feature importance plots
- Training report with metrics

---

## API Documentation

### Prediction Endpoint
```http
POST /predict
Content-Type: application/json
Authorization: Required

{
  "symptoms": ["fever", "cough"],
  "division": "Dhaka",
  "lat": 23.8103,
  "long": 90.4125
}
```

### Response
```json
{
  "success": true,
  "predictions": [
    {
      "disease": "Flu",
      "confidence": 0.78,
      "probability": "78.0%",
      "regional_influence": 0.15
    }
  ]
}
```

---

## Database Schema

### Core Tables

- **user_profiles** - User account information
- **predictions** - Disease prediction records
- **location_insights** - Regional disease trends
- **doctors** - Healthcare provider profiles
- **appointments** - Scheduled consultations
- **user_subscriptions** - Plan subscriptions
- **system_settings** - System configuration

---

## Deployment

### Local Development
```bash
python app.py
# Server runs on http://localhost:5000
```

### Production (Render)
1. Push to GitHub
2. Connect to Render
3. Set environment variables
4. Deploy with Gunicorn

### Production Checklist
- ✅ Strong `FLASK_SECRET_KEY`
- ✅ HTTPS enabled
- ✅ Database backups configured
- ✅ Error logging enabled
- ✅ Security headers set
- ✅ Rate limiting configured

---

## Contributing

Follow PEP 8 style conventions, include docstrings, and submit PRs with clear descriptions.

---

## Troubleshooting

### Supabase Connection Error
```bash
pip install --upgrade urllib3 certifi
```

### Model Not Found
```bash
python train_model.py
```

### Port Already in Use
```bash
lsof -ti:5000 | xargs kill -9
```

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Support

- **Author:** Abdullah Hossien (itzabd)
- **GitHub:** [@itzabd](https://github.com/itzabd)
- **Issues:** [GitHub Issues](https://github.com/itzabd/GloHealth_Ai/issues)

---

**Version:** 1.0.0 | **Status:** Active Development | **Updated:** April 24, 2026
