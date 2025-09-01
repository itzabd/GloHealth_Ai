GloHealth AI

GloHealth AI is a full-stack web application for healthcare management and AI-powered disease prediction. Built using Flask for the backend and Supabase for database and authentication, the system enables users to manage profiles, doctors, appointments, and receive predictive insights about potential diseases based on self-reported symptoms.

Project Overview / Motivation

Accurate disease identification at an early stage is critical for timely treatment and improved health outcomes. However, many individuals lack access to quick medical consultation or rely on inconsistent information sources when interpreting symptoms. This raises a primary research question:

How can a machine learning model effectively predict diseases based on a limited set of user-reported symptoms?

At the same time, healthcare professionals and public health authorities need tools that can highlight emerging health trends within communities to better allocate resources and respond to potential outbreaks. This leads to a second key question:

How can a system be designed to generate actionable insights for public health authorities while maintaining user privacy?

Advances in machine learning have enabled the development of systems that can predict likely diseases from basic symptom inputs. This project presents an AI-powered health symptom reporting system, which:

Systematically evaluates multiple machine learning models—including Random Forest, XGBoost, and Support Vector Machines—on publicly available datasets to identify the most reliable approach.

Deploys the final model through a web application, allowing users to input symptoms and receive instant predictions.

Provides visual explanations of which features influence predictions for transparency.

Integrates geographical and temporal data to improve prediction accuracy and highlight seasonal or regional disease trends.

By combining predictive accuracy with transparency and trend analysis, this system benefits both individuals seeking quick guidance and public health authorities monitoring evolving health challenges.

Features
User Management

Register, login, and manage user profiles

Admin dashboard for editing or deleting users

Role-based access control

Doctor Management

Add, edit, or remove doctors

Store doctor details: name, specialty, hospital, division, district, consultation fee, availability, and contact

Appointment Management

Schedule appointments between users and doctors

Track appointment status (pending, confirmed, completed) and payment status (unpaid, paid)

Admin dashboard to view, edit, or delete appointments

AI-Powered Disease Prediction

Input symptoms to receive predicted diseases

Compare multiple machine learning models for optimal accuracy

Visual explanations of model feature importance

Analyze seasonal and regional disease trends

Authentication

Supabase-based authentication

Admin-only routes for sensitive operations

Session management and role-based access

UI

Responsive admin dashboard

Multilingual support (English / Bangla)

Clean and modern interface with Bootstrap 5

Technology Stack

Backend: Python, Flask

Frontend: HTML, CSS, Bootstrap 5, Jinja2 templates

Database: PostgreSQL via Supabase

Authentication & Authorization: Supabase Auth

AI Models: Random Forest, XGBoost, Support Vector Machines

Hosting: Local / WSGI-compatible server

Installation
1. Clone the repository
git clone https://github.com/yourusername/glohealth-ai.git
cd glohealth-ai

2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows

3. Install dependencies
pip install -r requirements.txt

4. Configure environment variables

Create a .env file in the project root with Supabase credentials:

SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-service-key
FLASK_SECRET_KEY=your-secret-key

5. Run the application
flask run


Open http://localhost:5000 in your browser.

Database Setup
Supabase Tables

auth.users: Managed by Supabase Auth

user_profiles: Stores user info (id, full_name, email, address, division, city, postal_code)

doctors: Stores doctor details (id, name, specialty, hospital, division, district, consultation_fee, contact, availability)

appointments: Stores appointments (id, user_id, doctor_id, scheduled_time, status, payment_status, created_at)

Usage

Users register and login to the system.

Admin can manage users, doctors, and appointments through the dashboard.

Users can input symptoms to get AI-based disease predictions.

The system also provides trend analysis for healthcare monitoring.

Contributing

Fork the repository

Create a feature branch: git checkout -b feature/my-feature

Commit your changes: git commit -m "Add some feature"

Push to branch: git push origin feature/my-feature

Open a Pull Request

License

This project is licensed under the MIT License – see the LICENSE
 file for details.
