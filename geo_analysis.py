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

    def load_geo_data(self):
        """Load and process data with geo information."""
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
        full_df = pd.concat([train_df, test_df])

        # Generate mock coordinates if not available
        if 'lat' not in full_df.columns:
            full_df['lat'] = full_df['zip'].apply(lambda x: float(str(x)[:2]) + np.random.uniform(0, 1))
            full_df['long'] = full_df['zip'].apply(lambda x: float(str(x)[2:]) - 100 + np.random.uniform(-1, 1))

        return full_df

    def predict_diseases(self, df):
        """Add predicted diseases to dataframe."""
        X = df[self.feature_cols].fillna(0)
        df['predicted'] = self.le.inverse_transform(self.model.predict(X))
        df['confidence'] = np.max(self.model.predict_proba(X), axis=1)
        return df

    def create_disease_map(self, df):
        """Generate interactive disease cluster map."""
        disease_map = folium.Map(location=self.usa_center, zoom_start=self.zoom_level)

        # Add heatmap first (renders below markers)
        HeatMap(df[['lat', 'long', 'confidence']].values.tolist(),
                name="Case Density").add_to(disease_map)

        # Add clustered markers
        for disease in df['predicted'].unique():
            cluster = MarkerCluster(name=disease)
            subset = df[df['predicted'] == disease]

            for _, row in subset.iterrows():
                popup = f"""
                <b>{row['predicted']}</b><br>
                Confidence: {row['confidence']:.1%}<br>
                ZIP: {row['zip']}<br>
                Date: {row['timestamp']}
                """
                cluster.add_child(
                    folium.Marker(
                        location=[row['lat'], row['long']],
                        popup=popup,
                        icon=folium.Icon(color=self._get_disease_color(row['predicted']))
                    )
                )
            disease_map.add_child(cluster)

        folium.LayerControl().add_to(disease_map)
        return disease_map

    def plot_seasonal_trends(self, df):
        """Generate monthly disease frequency plots."""
        df['month'] = pd.to_datetime(df['timestamp']).dt.month_name()
        monthly_counts = df.groupby(['predicted', 'month']).size().unstack().fillna(0)

        plt.figure(figsize=(14, 8))
        sns.heatmap(monthly_counts, cmap="YlOrRd", annot=True, fmt='g')
        plt.title('Disease Cases by Month')
        plt.ylabel('Disease')
        plt.xlabel('Month')
        plt.tight_layout()
        return plt

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

    def _get_disease_color(self, disease):
        """Get consistent colors for diseases."""
        color_map = {
            'Diabetes': 'red',
            'Hypertension': 'blue',
            'Asthma': 'green',
            'Flu': 'orange',
            'COVID-19': 'purple'
        }
        return color_map.get(disease, 'gray')

    def run_analysis(self):
        """Execute full analysis pipeline."""
        print("Starting geo-temporal analysis...")

        try:
            df = self.load_geo_data()
            df = self.predict_diseases(df)

            # Generate visualizations
            self.create_disease_map(df).save('results/disease_clusters.html')
            self.plot_seasonal_trends(df).savefig('results/seasonal_trends.png')
            plt.close()

            # Update Supabase
            if self.update_supabase_insights(df):
                print("✅ Successfully updated Supabase")
            else:
                print("⚠️ Supabase update had some issues")

            print("Analysis completed successfully!")

        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")

    def calculate_regional_prevalence(self, df):
        """Calculate disease prevalence by division."""
        # Group by division and disease
        regional_stats = df.groupby(['division', 'predicted']) \
            .agg(count=('predicted', 'size'),
                 avg_confidence=('confidence', 'mean')) \
            .reset_index()

        # Normalize counts to get prevalence scores
        division_totals = regional_stats.groupby('division')['count'].sum()
        regional_stats['prevalence_score'] = regional_stats.apply(
            lambda row: row['count'] / division_totals[row['division']],
            axis=1
        )

        return regional_stats

    def update_supabase_insights(self, df):
        """Update Supabase with enhanced regional insights."""
        regional_stats = self.calculate_regional_prevalence(df)

        records = []
        for _, row in regional_stats.iterrows():
            records.append({
                'division': row['division'],
                'disease': row['predicted'],
                'confidence_score': float(row['avg_confidence']),
                'prevalence_score': float(row['prevalence_score']),
                'last_updated': datetime.now().isoformat()
            })

        # Batch upsert
        response = self.supabase.table('location_insights').upsert(
            records,
            on_conflict='division,disease'
        ).execute()

        return response

if __name__ == "__main__":
    analyzer = GeoAnalyzer()
    analyzer.run_analysis()