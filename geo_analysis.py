import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from supabase import create_client, Client
from datetime import datetime
import seaborn as sns
import warnings
from time import sleep

warnings.filterwarnings('ignore')


class GeoAnalyzer:
    def __init__(self):
        # Initialize Supabase client
        self.supabase: Client = create_client(
            supabase_url="https://qmktyfkebpjtihxmfbgp.supabase.co",
            supabase_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFta3R5ZmtlYnBqdGloeG1mYmdwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM2NzA3NTYsImV4cCI6MjA2OTI0Njc1Nn0.cAqokDfgN3PgHTQzyW-bPELgJlm3--a-O_Q97SFeTEk"
        )

        # Load model artifacts
        self.model = joblib.load('results/production_model.joblib')
        self.le = joblib.load('results/label_encoder.joblib')
        self.feature_cols = joblib.load('results/feature_columns.joblib')

        # Geo configuration
        self.usa_center = [37.0902, -95.7129]
        self.zoom_level = 4

    def _ensure_table_exists(self):
        """Ensure the table exists with correct structure."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try a simple query to check table existence
                self.supabase.table('location_insights').select("zip_code").limit(1).execute()
                return True
            except Exception as e:
                print(f"Attempt {attempt + 1}: Table not found, creating...")
                try:
                    # Create table via RPC
                    self.supabase.rpc('create_location_insights_table').execute()
                    sleep(2)  # Wait for table creation
                except Exception as create_error:
                    print(f"Table creation failed: {create_error}")
                    if attempt == max_retries - 1:
                        raise RuntimeError("Failed to create table after multiple attempts")
        return False

    def update_supabase_insights(self, df):
        """Update Supabase with aggregated location insights."""
        try:
            if not self._ensure_table_exists():
                raise RuntimeError("Could not verify or create table")

            # Prepare data with EXACT column names
            records = []
            for (zip_code, disease), group in df.groupby(['zip', 'predicted']):
                records.append({
                    'zip_code': str(zip_code),
                    'disease': str(disease),
                    'confidence_score': float(group['confidence'].mean()),
                    'lat': float(group['lat'].iloc[0]),
                    'lon': float(group['long'].iloc[0])
                })

            # Batch upsert with error handling
            response = self.supabase.table('location_insights').upsert(
                records,
                on_conflict='zip_code,disease'
            ).execute()

            # Verify successful update
            if len(records) > 0 and not response.data:
                raise RuntimeError("No data was inserted/updated")

            return True

        except Exception as e:
            print(f"⚠️ Supabase update failed: {str(e)}")
            print(f"⚠️ Attempted to insert {len(records)} records")
            if records:
                print("⚠️ First record example:", records[0])
            return False

    # [Keep all other methods exactly the same]


if __name__ == "__main__":
    analyzer = GeoAnalyzer()
    analyzer.run_analysis()