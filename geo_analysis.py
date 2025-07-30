import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from supabase import create_client
import os
from datetime import datetime
import seaborn as sns


class GeoAnalyzer:
    def __init__(self):
        # Load model artifacts
        self.model = joblib.load('results/production_model.joblib')
        self.le = joblib.load('results/label_encoder.joblib')
        self.feature_cols = joblib.load('results/feature_columns.joblib')

        # Supabase client
        self.supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )

        # Geo configuration
        self.usa_center = [37.0902, -95.7129]
        self.zoom_level = 4

    def load_geo_data(self):
        """Load data with geo information from processed CSVs."""
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")

        # Combine datasets
        full_df = pd.concat([train_df, test_df])

        # Generate mock coordinates if not available
        if 'lat' not in full_df.columns:
            full_df['lat'] = [float(str(zip)[:2]) + np.random.uniform(0, 1) for zip in full_df['zip']]
            full_df['long'] = [float(str(zip)[2:]) - 100 + np.random.uniform(-1, 1) for zip in full_df['zip']]

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

        # Cluster markers by disease type
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

        # Add heatmap overlay
        heat_data = df[['lat', 'long', 'confidence']].values.tolist()
        HeatMap(heat_data, name="Case Density").add_to(disease_map)

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

    def update_supabase_insights(self, df):
        """Update Supabase with aggregated location insights."""
        try:
            # Group by ZIP and disease
            insights = df.groupby(['zip', 'predicted']).agg({
                'confidence': 'mean',
                'lat': 'first',
                'long': 'first'
            }).reset_index()

            # Upsert to Supabase
            for _, row in insights.iterrows():
                self.supabase.table('location_insights').upsert({
                    'zip_code': row['zip'],
                    'disease': row['predicted'],
                    'avg_confidence': float(row['confidence']),
                    'latitude': float(row['lat']),
                    'longitude': float(row['long']),
                    'last_updated': datetime.now().isoformat()
                }, on_conflict='zip_code,disease').execute()

            return True
        except Exception as e:
            print(f"Error updating Supabase: {e}")
            return False

    def _get_disease_color(self, disease):
        """Assign consistent colors to diseases."""
        colors = {
            'Diabetes': 'red',
            'Hypertension': 'blue',
            'Asthma': 'green',
            'Flu': 'orange',
            'COVID-19': 'purple'
        }
        return colors.get(disease, 'gray')

    def run_analysis(self):
        """Execute full geo-temporal analysis pipeline."""
        print("Loading and processing data...")
        df = self.load_geo_data()
        df = self.predict_diseases(df)

        print("Generating visualizations...")
        # Disease map
        disease_map = self.create_disease_map(df)
        disease_map.save('results/disease_clusters.html')

        # Seasonal trends
        seasonal_plot = self.plot_seasonal_trends(df)
        seasonal_plot.savefig('results/seasonal_trends.png')
        seasonal_plot.close()

        # Update Supabase
        if self.update_supabase_insights(df):
            print("Successfully updated Supabase insights")
        else:
            print("Failed to update Supabase")

        print("Analysis complete!")


if __name__ == "__main__":
    analyzer = GeoAnalyzer()
    analyzer.run_analysis()