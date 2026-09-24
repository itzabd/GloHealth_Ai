# Product Requirements & Specification Document (PRD)

## Project: GloHealth AI
**Version:** 1.0.0  
**Application Type:** Full-Stack AI-Powered Healthcare Web Application  
**Primary Target URL:** `http://localhost:5000/`  
**Tech Stack:** Python 3.11, Flask 3.1.1, Supabase (PostgreSQL & Auth), scikit-learn, Jinja2, Bootstrap 5, Vanilla JavaScript  

---

## 1. Executive Summary & Product Vision

**GloHealth AI** is an intelligent, full-stack healthcare web platform designed to democratize clinical symptom triage, connect patients with verified medical practitioners, and provide epidemiological geospatial disease tracking across Bangladesh's administrative divisions.

The application integrates machine learning predictive models (trained Random Forest classifier) with real-time location-based epidemiological weighting and seasonal adjustments to deliver accurate, explainable disease predictions and direct care pathways.

### Core Objectives:
1. **Instant Symptom Triage**: Enable users to report multiple categorized symptoms and receive instant differential diagnosis predictions with confidence scoring.
2. **Explainable AI Insights**: Deliver transparent predictions highlighting key contributory symptoms and severity assessments.
3. **Seamless Doctor Consultation Booking**: Provide a searchable, filterable directory of specialized doctors and appointment scheduling with quota-based checkup points.
4. **Epidemiological & Geospatial Tracking**: Visualize regional disease distribution, seasonal patterns, and active outbreak clusters.
5. **Subscription-Based Care Management**: Support tiered health plans (Basic, Premium, Ultimate) offering bundled doctor consultations.
6. **Comprehensive Admin Governance**: Provide healthcare administrators with granular oversight of users, doctors, appointments, system settings, and audit logs.
7. **Bilingual Accessibility**: Full bilingual support in English (`en`) and Bengali (`bn`).

---

## 2. User Roles & Personas

| Role | Description | Access Rights & Boundaries |
|------|-------------|----------------------------|
| **Visitor / Guest** | Unauthenticated public user exploring the platform | Access to Landing page, public features, doctor browsing, plan comparison, and login/registration modals. Prompted to authenticate for booking or symptom triage. |
| **Patient / Registered User** | Authenticated healthcare seeker | Full access to symptom checker (`/prediction`), ML prediction results (`/results`), personal dashboard (`/dashboard`), appointment booking (`/book_appointment`), appointments list (`/appointments`), and subscription management (`/plans`). |
| **Healthcare Administrator** | Clinical operations and system administrator | Full access to Admin Portal (`/admin/dashboard`), user management (`/admin/users`), doctor registry CRUD (`/admin/doctors`), appointment schedule management (`/admin/appointments`), system settings (`/admin/settings`), and audit logs (`/admin/export_audit`). |

---

## 3. Information Architecture & Route Specification

| Route / Endpoint | HTTP Methods | Access Level | Description | Key UI Components & Interactions |
|------------------|--------------|--------------|-------------|-----------------------------------|
| `/` | `GET` | Public | Homepage / Landing | Hero banner, feature highlights, quick-check CTA, stats counter, header navigation, auth trigger buttons. |
| `/login` | `GET`, `POST` | Public | User Login Page / Modal | Email and password input form, "Remember Me" toggle, links to signup, error banner for invalid credentials. |
| `/signup` | `GET`, `POST` | Public | User Registration | Full name, email, password, division selector, district, postal code, submit button. |
| `/logout` | `GET` | Authenticated | User Logout | Terminates Flask session and Supabase auth token, redirects to `/`. |
| `/dashboard` | `GET` | Authenticated | Patient Dashboard | Summary metrics (total assessments, upcoming appointments, active plan), recent prediction cards, quick actions. |
| `/prediction` | `GET` | Authenticated | Clinical Symptom Checker | Categorized symptom checkboxes/chips, symptom search bar, division & demographics inputs, submit CTA. |
| `/predict` | `POST` | Authenticated | ML Prediction API | Processes selected symptoms, applies location boosting, returns top predictions, confidence, and advice. |
| `/results` | `GET` | Authenticated | Prediction Results View | Primary disease card, confidence badge, feature importance breakdown, emergency guidance, "Book Doctor" CTA. |
| `/doctors` | `GET` | Public / User | Doctor Directory | Search input, filter by medical specialty, filter by division, doctor cards with fee and availability. |
| `/book_appointment/<doctor_id>` | `GET`, `POST` | Authenticated | Appointment Booking | Selected doctor summary, date picker, time slot selector, reason for visit, payment option / plan point usage. |
| `/appointments` | `GET` | Authenticated | My Appointments List | Table/cards of scheduled consultations, status chips (Pending, Confirmed, Completed), Cancel button. |
| `/plans` | `GET` | Public / User | Membership Plans | Tier cards (Basic, Premium, Ultimate), feature comparisons, "Subscribe" CTA buttons. |
| `/plan/<plan_name>` | `GET` | Public / User | Plan Details Page | Detailed breakdown of specific tier benefits, perks, and consultation allowances. |
| `/subscribe/<plan_name>` | `POST` | Authenticated | Plan Activation | Activates selected plan, updates checkup points, redirects to confirmation. |
| `/cancel_subscription/<sub_id>` | `POST` | Authenticated | Plan Cancellation | Deactivates subscription, preserves accrued history. |
| `/geo_insights` | `GET` | Authenticated | Geospatial Surveillance | Bangladesh division interactive map, disease prevalence heatmaps, outbreak metrics, divisional risk table. |
| `/set_language/<lang>` | `GET`, `POST` | Public | Language Switcher | Toggles interface language between English (`en`) and Bengali (`bn`), sets session and cookie. |
| `/admin/dashboard` | `GET` | Admin | Admin Overview | Platform KPI metrics, user growth, appointment statuses, quick links, audit log download. |
| `/admin/users` | `GET` | Admin | User Directory | Paginated list of registered users, role indicators, edit/delete actions. |
| `/edit_user/<user_id>` | `GET`, `POST` | Admin | Edit User Profile | Form to update name, email, division, admin status. |
| `/delete_user/<user_id>` | `GET`, `POST` | Admin | Remove User | Confirmation dialogue to safely delete user profile. |
| `/admin/doctors` | `GET` | Admin | Doctor Management | Doctor directory table, filter by specialty/division, "Add New Doctor" button. |
| `/admin/doctors/add` | `GET`, `POST` | Admin | Add Doctor | Form: name, specialty, division, district, hospital, fee, availability, contact phone. |
| `/admin/doctors/edit/<doctor_id>` | `GET`, `POST` | Admin | Edit Doctor | Form prefilled with doctor details to modify information. |
| `/admin/doctors/delete/<doctor_id>` | `POST` | Admin | Delete Doctor | Removes doctor from registry and updates active appointments. |
| `/admin/appointments` | `GET` | Admin | Appointment Management | Table of all bookings across doctors, filter by status and date, status changer dropdown. |
| `/admin/appointments/add` | `GET`, `POST` | Admin | Manual Appointment Booking | Admin booking tool to schedule appointment for any registered user and doctor. |
| `/admin/appointments/edit/<id>` | `GET`, `POST` | Admin | Edit Appointment | Reschedule time, update status (pending, confirmed, completed, cancelled), update payment status. |
| `/admin/appointments/delete/<id>` | `POST` | Admin | Delete Appointment | Cancels and removes appointment record. |
| `/admin/settings` | `GET`, `POST` | Admin | System Settings | Platform configuration: Site name, support email, base consultation fee. |
| `/admin/export_audit` | `GET` | Admin | Audit Logs Export | Streams downloadable CSV containing platform audit trails. |

---

## 4. Detailed Feature Specifications & User Flows

### Feature 1: Landing Page & Navigation
- **Path:** `/`
- **User Story:** As an unauthenticated or authenticated user, I want to view a modern, responsive landing page to understand the value of GloHealth AI, initiate health assessments, browse doctors, and navigate between sections.
- **Expected Elements & Behaviors:**
  - **Header / Navbar:** Logo branded as "GloHealth AI", navigation links (`Home`, `Symptom Checker`, `Doctors`, `Health Plans`, `Geospatial Insights`, `Language Toggle`).
  - **Auth Controls:** If unauthenticated: "Login" and "Get Started" buttons trigger authentication modals or redirect to `/login` and `/signup`. If authenticated: User profile badge, link to `/dashboard`, and "Logout" button.
  - **Hero Section:** Clear title, subtitle describing AI triage in Bangladesh, primary CTA "Check Symptoms Now" pointing to `/prediction`, secondary CTA "Find Doctors" pointing to `/doctors`.
  - **Trust & Stats Bar:** Real-time counters showing total assessments performed, verified medical specialists, and covered districts.
  - **Feature Showcase:** Grid cards detailing AI symptom diagnosis, geospatial heatmaps, doctor booking, and multilingual support.
  - **Language Selector:** Dropdown or toggle button switching between `English` and `বাংলা`. Upon click, fires `/set_language/<lang>` and reloads the page with translated strings.
  - **Footer:** Informational links, copyright notice, medical disclaimer ("GloHealth AI provides preliminary health assessments and is not a substitute for professional clinical advice").

### Feature 2: User Authentication & Account Management
- **Paths:** `/login`, `/signup`, `/logout`
- **User Story:** As a user, I want to securely register an account, sign in using my email and password, and sign out when finished.
- **Expected Elements & Behaviors:**
  - **Registration Form:**
    - Inputs: Full Name (text, required), Email Address (email, required), Password (password, min 6 characters, required), Division (select dropdown: Dhaka, Chattogram, Rajshahi, Khulna, Barishal, Sylhet, Rangpur, Mymensingh), District (text), Postal Code (text).
    - Validation: Live or submit-time validation for valid email syntax and non-empty mandatory fields.
    - Submit: Clicking "Create Account" creates the user in Supabase Auth, initializes a record in `user_profiles`, creates a free `user_subscriptions` tier with 3 free checkup points, logs the user in, and redirects to `/dashboard`.
    - Error Handling: Displays clear alert message if the email is already registered or password criteria are unmet.
  - **Login Form:**
    - Inputs: Email Address (email, required), Password (password, required), "Remember Me" checkbox.
    - Validation & Execution: Authenticates against Supabase Auth. On success, establishes Flask session via `Flask-Login` and redirects to `/dashboard` or previously requested protected route.
    - Error Handling: Invalid email or password displays warning message without revealing which credential was invalid.
  - **Logout Flow:**
    - Clicking "Logout" terminates the user session, clears cookies, and redirects the browser to `/` with a logged-out notification.

### Feature 3: Clinical Symptom Assessment & Machine Learning Prediction
- **Paths:** `/prediction`, `/predict`, `/results`
- **User Story:** As a patient, I want to select the symptoms I am experiencing from categorized options, specify my location, and receive an instant, probabilistic disease prediction with clinical recommendations.
- **Expected Elements & Behaviors:**
  - **Symptom Selection Form (`/prediction`):**
    - Search Bar: Quick real-time filter to find symptoms by name (e.g., "fever", "cough", "fatigue").
    - Category Tabs / Accordions:
      - *General*: High fever, fatigue, chills, body ache, weight loss, malaise.
      - *Respiratory*: Cough, shortness of breath, chest tightness, sore throat, runny nose.
      - *Gastrointestinal*: Nausea, vomiting, abdominal pain, diarrhea, loss of appetite.
      - *Neurological*: Headache, dizziness, loss of balance, sensitivity to light.
      - *Dermatological*: Skin rash, itching, yellowing of skin/eyes.
      - *Musculoskeletal*: Joint pain, muscle weakness, neck stiffness.
    - Demographics & Location:
      - Division selector (pre-filled from user profile if authenticated).
      - Optional GPS auto-detection button to populate latitude and longitude.
    - Submit CTA: "Analyze Symptoms" button (disabled if fewer than 1 symptom is selected).
  - **Prediction Engine Execution (`/predict`):**
    - Converts selected symptoms into binary vector matching trained model feature columns (`feature_columns.joblib`).
    - Random Forest model evaluates symptom vector to generate baseline class probabilities.
    - Applies regional boost multiplier based on historical disease prevalence in `location_insights` for the user's division.
    - Computes top 3 predicted diseases with normalized confidence percentages.
    - Persists prediction run in `predictions` table linked to user ID.
  - **Results Presentation (`/results` or result modal):**
    - Top predicted disease name in prominent header (e.g. "Dengue Fever - 87% Confidence").
    - Severity Badge: Normal / Moderate / Urgent triage level.
    - Clinical Explanation: Feature importance graph or list showing top 3 symptoms that influenced the prediction (e.g. high fever, retro-orbital pain, joint pain).
    - Medical Advisory: Clear next steps (self-care advice, when to visit urgent care).
    - Quick Action: "Consult a Doctor" button linking directly to filtered `/doctors` matching the relevant medical specialty.

### Feature 4: Doctor Directory & Consultation Booking
- **Paths:** `/doctors`, `/book_appointment/<doctor_id>`, `/appointments`
- **User Story:** As a patient, I want to search for verified healthcare providers, filter by specialty and geographic division, and schedule an appointment.
- **Expected Elements & Behaviors:**
  - **Doctor Directory (`/doctors`):**
    - Search bar: Filter doctors by name or hospital.
    - Filters: Dropdown for Specialty (General Physician, Cardiologist, Dermatologist, Pulmonologist, Neurologist, Pediatrician, etc.) and Division.
    - Doctor Card: Shows doctor name, qualifications, specialty badge, affiliated hospital/clinic, division/district, consultation fee (BDT), available consultation hours, and "Book Appointment" CTA button.
  - **Booking Interface (`/book_appointment/<doctor_id>`):**
    - Doctor details banner displaying doctor profile and regular fee.
    - Date Picker: Calendar input to choose appointment date (restricted to future dates).
    - Time Slot Selection: Available slots based on doctor's schedule (e.g., "10:00 AM - 10:30 AM", "04:00 PM - 04:30 PM").
    - Reason / Symptoms: Textarea to provide reason for visit or attach recent prediction ID.
    - Payment / Points Selector:
      - Option A: Use active subscription checkup points (deducts 1 point, fee = 0 BDT).
      - Option B: Pay consultation fee directly (redirects to mock payment confirmation).
    - Confirmation: Submitting creates record in `appointments` table with status `pending` or `confirmed`.
  - **Appointments List (`/appointments`):**
    - Table/Cards displaying: Doctor Name, Specialty, Hospital, Appointment Date & Time, Fee / Points Used, Status badge (`Pending`, `Confirmed`, `Completed`, `Cancelled`).
    - Cancel Action: Allows user to cancel pending/future appointments with confirmation dialog.

### Feature 5: Geospatial Health Insights & Outbreak Surveillance
- **Path:** `/geo_insights`
- **User Story:** As a patient or public health researcher, I want to explore disease distribution across Bangladesh to understand local outbreak trends.
- **Expected Elements & Behaviors:**
  - **Interactive Map:** Leaflet/Folium map rendering 8 divisions of Bangladesh (Dhaka, Chattogram, Rajshahi, Khulna, Barishal, Sylhet, Rangpur, Mymensingh) with colored circle markers / polygons representing case density.
  - **Disease Filter Dropdown:** Select specific disease (e.g. "Dengue", "COVID-19", "Typhoid", "Malaria", or "All Diseases").
  - **Divisional Breakdown Cards:** Displays total reported cases, dominant symptoms, and outbreak risk level per division.
  - **Temporal & Seasonal Chart:** Shows disease trends across calendar months to highlight seasonal peaks (e.g., monsoon dengue surge).

### Feature 6: Subscription Plans & Health Benefits
- **Paths:** `/plans`, `/plan/<plan_name>`, `/subscribe/<plan_name>`, `/cancel_subscription/<sub_id>`
- **User Story:** As a user, I want to compare subscription tiers and enroll in a plan that provides free consultation points and priority care.
- **Expected Elements & Behaviors:**
  - **Tier Comparison Grid (`/plans`):**
    - **Basic Plan**: Free tier, includes 3 symptom checks, standard doctor directory access, community support.
    - **Premium Plan**: 499 BDT/month, includes 5 doctor consultation points, priority booking, detailed AI explainability, family profile support.
    - **Ultimate Plan**: 999 BDT/month, unlimited AI checks, 12 doctor consultation points, instant specialist triage, 24/7 tele-support.
  - **Subscription Activation (`/subscribe/<plan_name>`):**
    - Displays selected plan details, total price, and allocated checkup points.
    - Clicking "Confirm Subscription" updates active subscription in `user_subscriptions` and adds checkup points to user balance.
  - **Subscription Cancellation (`/cancel_subscription/<sub_id>`):**
    - Allows user to cancel active renewal while retaining points until the end of billing period.

### Feature 7: Administrative Governance Portal
- **Paths:** `/admin/dashboard`, `/admin/users`, `/admin/doctors`, `/admin/appointments`, `/admin/settings`, `/admin/export_audit`
- **User Story:** As a system administrator, I want a centralized dashboard to oversee system health, manage doctors and appointments, modify platform settings, and export compliance audit logs.
- **Expected Elements & Behaviors:**
  - **Admin Navigation & Access Control:** Restricts access to users where `is_admin == true`. Unauthorized users are redirected to `/login` or given a 403 Forbidden error.
  - **Admin Dashboard (`/admin/dashboard`):** KPI summary widgets (Total Users, Active Doctors, Total Appointments, Completed Consultations), recent system activity feed, audit download button.
  - **User Management (`/admin/users`):** List registered accounts, search by email/name, edit user details (`/edit_user/<id>`), and delete user (`/delete_user/<id>`).
  - **Doctor Management (`/admin/doctors`):**
    - Add Doctor (`/admin/doctors/add`): Form to add new medical specialist with hospital, division, fee, and availability.
    - Edit Doctor (`/admin/doctors/edit/<id>`): Form to modify doctor data.
    - Delete Doctor (`/admin/doctors/delete/<id>`): Action to remove doctor.
  - **Appointment Management (`/admin/appointments`):**
    - Add Appointment (`/admin/appointments/add`): Schedule appointment manually.
    - Edit Appointment (`/admin/appointments/edit/<id>`): Change date, time, status (`pending`, `confirmed`, `completed`, `cancelled`), and payment status (`unpaid`, `paid`).
    - Delete Appointment (`/admin/appointments/delete/<id>`).
  - **Settings (`/admin/settings`):** Form to update site title, support email, and platform commission / base fee.
  - **Audit Export (`/admin/export_audit`):** Initiates direct CSV file download containing platform audit trail records.

### Feature 8: Bilingual Internationalization (EN / BN)
- **Path:** `/set_language/<lang>`
- **User Story:** As a Bengali or English speaker, I want the web application to display in my preferred language.
- **Expected Elements & Behaviors:**
  - Supports codes: `en` (English) and `bn` (বাংলা / Bengali).
  - Toggling language sends `GET`/`POST` to `/set_language/<lang>`, updating session variable `session['lang']` and setting a persistent cookie `lang` (max-age 1 year).
  - Templates render translated labels for navigation items, symptom names, assessment results, buttons, and system notices.

---

## 5. Non-Functional Requirements & Test Assertions

| Area | Requirement | Verification Method |
|------|-------------|---------------------|
| **Response Time** | Page loads within < 1.5s on local server; ML inference completes in < 500ms | Performance assertion on `/predict` and `/doctors` routes |
| **Authentication Security** | Protected routes (`/prediction`, `/dashboard`, `/appointments`, `/geo_insights`) must redirect unauthenticated visitors to `/login` | HTTP 302 / redirect assertions |
| **Admin Authorization** | Admin routes (`/admin/*`) must reject non-admin users with 403 Forbidden or redirect | Role-based authorization test |
| **Input Validation** | Registration and symptom forms reject invalid email syntax or empty payloads | Form error assertion |
| **Responsive Layout** | Layout adapts seamlessly to Desktop (1920x1080), Tablet (768x1024), and Mobile (375x667) | Viewport responsiveness testing |
| **Cross-Browser Compatibility** | Modern Chromium, Gecko, and WebKit rendering engines | Playwright / TestSprite browser engine checks |

---

## 6. End-to-End Test Scenarios for TestSprite

### Scenario 1: Landing Page & Unauthenticated Exploration
1. Navigate to `/`.
2. Verify page title contains "GloHealth AI".
3. Verify presence of Hero banner, "Check Symptoms", and "Find Doctors" buttons.
4. Verify navigation links: Home, Symptom Checker, Doctors, Plans, Language Toggle.
5. Click on "Find Doctors" and verify navigation to `/doctors`.
6. Filter doctors by specialty "Cardiology" and verify doctor cards update.
7. Click "Language Toggle" to switch to Bengali (`bn`) and verify header translation.

### Scenario 2: User Registration & Onboarding
1. Navigate to `/signup`.
2. Fill registration form:
   - Full Name: `Test Patient`
   - Email: `testpatient_<timestamp>@example.com`
   - Password: `Password123!`
   - Division: `Dhaka`
3. Click "Create Account" submit button.
4. Verify user is redirected to `/dashboard` or home with active session.
5. Verify welcome message displays user name and default free plan status.

### Scenario 3: Symptom Assessment & AI Prediction
1. Navigate to `/prediction` as authenticated user.
2. Select symptoms: `High Fever`, `Fatigue`, `Body Ache`, `Joint Pain`.
3. Ensure division is set to `Dhaka`.
4. Click "Analyze Symptoms" / "Predict".
5. Verify redirection to results view or results modal appearance.
6. Verify predicted disease name, confidence score percentage badge, and clinical recommendation.
7. Verify "Consult Doctor" CTA is visible and clickable.

### Scenario 4: Doctor Appointment Booking Flow
1. Navigate to `/doctors`.
2. Choose a doctor card and click "Book Appointment".
3. Verify navigation to `/book_appointment/<doctor_id>`.
4. Select appointment date (future date) and choose time slot.
5. Select payment method ("Use Checkup Points" or "Direct Payment").
6. Click "Confirm Booking".
7. Verify redirection to `/appointments` with newly booked appointment visible under `Pending` status.

### Scenario 5: Health Plans & Subscription Activation
1. Navigate to `/plans`.
2. Verify display of Basic, Premium, and Ultimate pricing cards.
3. Click "Choose Premium" on Premium Plan card.
4. Verify navigation or confirmation modal for `/subscribe/Premium Plan`.
5. Confirm subscription and verify points balance updates on dashboard.

### Scenario 6: Admin Portal Operations
1. Authenticate with admin account.
2. Navigate to `/admin/dashboard`.
3. Verify KPI summary cards render (Total Users, Active Doctors, Appointments).
4. Navigate to `/admin/doctors`.
5. Click "Add Doctor", fill form (Name: `Dr. Sarah Rahman`, Specialty: `Dermatology`, Fee: `1000`, Division: `Dhaka`), and submit.
6. Verify doctor appears in the doctor list.
7. Navigate to `/admin/appointments` and update an appointment status to `Confirmed`.
8. Verify audit log export link `/admin/export_audit` returns a downloadable CSV.
