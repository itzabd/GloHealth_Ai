# AI-Powered Health Symptom Reporting System

## Project Overview / Motivation

Accurate disease identification at an early stage is critical for timely treatment and improved health outcomes. However, many individuals lack access to quick medical consultation or rely on inconsistent information sources when interpreting symptoms. This raises a primary research question: 

> **How can a machine learning model effectively predict diseases based on a limited set of user-reported symptoms?**

At the same time, healthcare professionals and public health authorities need tools that can highlight emerging health trends within communities to better allocate resources and respond to potential outbreaks. This leads to a second key question:  

> **How can a system be designed to generate actionable insights for public health authorities while maintaining user privacy?**

Advances in machine learning have enabled the development of systems that can predict likely diseases from basic symptom inputs. This project presents an **AI-powered health symptom reporting system**, which:  

- Systematically evaluates multiple machine learning models—including **Random Forest**, **XGBoost**, and **Support Vector Machines**—on publicly available datasets to identify the most reliable approach.  
- Deploys the final model through a web application, allowing users to input symptoms and receive instant predictions.  
- Provides **visual explanations** of which features influence predictions for transparency.  
- Integrates **geographical and temporal data** to improve prediction accuracy and highlight seasonal or regional disease trends.  

By combining predictive accuracy with transparency and trend analysis, this system benefits both **individuals seeking quick guidance** and **public health authorities** monitoring evolving health challenges.

---

## Features

- **User-Facing:**
  - Submit symptoms to receive instant disease predictions.
  - View feature importance visualizations for transparency.
  - Explore seasonal and regional disease trends.

- **Admin Dashboard:**
  - Manage users, doctors, and appointments.
  - View aggregated disease trends and analytics.
  - Monitor system usage and predictive performance.

- **AI & Analytics:**
  - Model evaluation: Random Forest, XGBoost, SVM.
  - Predictive explanations using feature importance.
  - Geotemporal disease trend analysis.

---

## Tech Stack

- **Backend:** Python Flask
- **Database:** Supabase (PostgreSQL)
- **Machine Learning:** scikit-learn, XGBoost
- **Frontend:** HTML, CSS, Bootstrap
- **Deployment:** Render / Local Server

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-health-symptom-system.git
cd ai-health-symptom-system
