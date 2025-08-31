# Updated geo_analysis.py
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

        # Bangladesh center coordinates
        self.bd_center = [23.6850, 90.3563]
        self.zoom_level = 7

    def load_geo_data(self):
        """Load and process data with geo information from Supabase."""
        try:
            # Load predictions data from Supabase
            predictions = self.supabase.from_('predictions').select('*').execute()
            if not predictions.data:
                raise ValueError("No prediction data found in Supabase")

            df = pd.DataFrame(predictions.data)

            # Generate coordinates if not available (mock data for Bangladesh)
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                # Assign random coordinates within Bangladesh bounds
                df['latitude'] = np.random.uniform(20.5, 26.5, size=len(df))
                df['longitude'] = np.random.uniform(88.0, 92.5, size=len(df))

            return df

        except Exception as e:
            print(f"Error loading geo data: {str(e)}")
            raise

    def create_bangladesh_map(self, df):
        """Generate interactive disease cluster map for Bangladesh."""
        bd_map = folium.Map(location=self.bd_center, zoom_start=self.zoom_level, tiles='cartodbpositron')

        # Add Bangladesh boundary
        folium.GeoJson(
            'https://raw.githubusercontent.com/geohacker/bangladesh/master/bd-districts.geojson',
            name='Bangladesh Districts',
            style_function=lambda x: {'fillColor': '#ffff00', 'color': '#000000', 'weight': 0.5}
        ).add_to(bd_map)

        # Add heatmap
        HeatMap(
            df[['latitude', 'longitude', 'confidence']].values.tolist(),
            name="Case Density",
            radius=15,
            blur=10,
            max_zoom=1
        ).add_to(bd_map)

        # Add clustered markers
        marker_cluster = MarkerCluster(name="Cases").add_to(bd_map)

        for _, row in df.iterrows():
            popup = f"""
            <b>{row['top_prediction']}</b><br>
            Confidence: {row['confidence']:.1%}<br>
            Division: {row['division']}<br>
            Date: {row['timestamp']}
            """
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup,
                icon=folium.Icon(color=self._get_disease_color(row['top_prediction']))
            ).add_to(marker_cluster)

        folium.LayerControl().add_to(bd_map)
        return bd_map

    def plot_division_trends(self, df):
        """Generate disease frequency plots by division."""
        plt.figure(figsize=(14, 8))
        division_counts = df.groupby(['division', 'top_prediction']).size().unstack().fillna(0)
        sns.heatmap(division_counts, cmap="YlOrRd", annot=True, fmt='g')
        plt.title('Disease Cases by Division')
        plt.ylabel('Division')
        plt.xlabel('Disease')
        plt.tight_layout()
        return plt

    def run_analysis(self):
        """Execute full analysis pipeline for Bangladesh."""
        print("Starting Bangladesh geo-temporal analysis...")
        try:
            df = self.load_geo_data()

            # Generate visualizations
            self.create_bangladesh_map(df).save('results/bangladesh_disease_map.html')
            self.plot_division_trends(df).savefig('results/division_trends.png')
            plt.close()

            print("Analysis completed successfully!")
            return True
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return False

    def _get_disease_color(self, disease):
        """Get consistent colors for diseases."""
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


if __name__ == "__main__":
    analyzer = GeoAnalyzer()
    analyzer.run_analysis()