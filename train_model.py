import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import json
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class ModelTrainer:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                class_weight='balanced_subsample',
                random_state=42,
                max_depth=12,
                min_samples_split=3,
                max_features='log2',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                objective='multi:softmax',
                n_estimators=300,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                subsample=0.8
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=300,
                class_weight='balanced_subsample',
                random_state=42,
                max_depth=12,
                min_samples_split=3,
                max_features='log2',
                n_jobs=-1
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.le = LabelEncoder()
        self.valid_columns = None
        self.dropped_columns = []
        self.feature_columns = None

    def load_data(self):
        """Load and preprocess data with proper encoding."""
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")

        # Define geo-temporal columns to preserve
        geo_cols = ['zip', 'timestamp']

        # Store geo data separately
        self.geo_train = train_df[geo_cols].copy()
        self.geo_test = test_df[geo_cols].copy()

        # Encode labels consistently
        all_prognosis = pd.concat([train_df['prognosis'], test_df['prognosis']])
        self.le.fit(all_prognosis)

        # Prepare features EXCLUDING geo columns
        X_train = train_df.drop(['prognosis'] + geo_cols, axis=1)
        y_train = self.le.transform(train_df['prognosis'])

        X_test = test_df.drop(['prognosis'] + geo_cols, axis=1)
        y_test = self.le.transform(test_df['prognosis'])

        # Identify columns with at least one valid value
        self.valid_columns = X_train.columns[~X_train.isna().all()].tolist()
        self.dropped_columns = [col for col in X_train.columns if col not in self.valid_columns]

        if self.dropped_columns:
            print(f"\nDropping columns with all NaN values: {self.dropped_columns}")

        # Keep only valid columns
        X_train = X_train[self.valid_columns]
        X_test = X_test[[col for col in self.valid_columns if col in X_test.columns]]

        # Fill any remaining NaN values with 0 (assuming binary symptoms)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Store feature columns
        self.feature_columns = list(X_train.columns)

        print("\n=== Class Distribution ===")
        print("Training set:")
        print(Counter(self.le.inverse_transform(y_train)))
        print("\nTest set:")
        print(Counter(self.le.inverse_transform(y_test)))

        return X_train, X_test, y_train, y_test

    def handle_imbalance(self, X_train, y_train):
        """Handle class imbalance with adaptive resampling."""
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())

        print("\n=== Handling Class Imbalance ===")
        if min_samples < 5:
            print("Using combined SMOTE + Undersampling strategy")
            over = SMOTE(
                sampling_strategy='not minority',
                k_neighbors=min(5, min_samples - 1),
                random_state=42
            )
            under = RandomUnderSampler(
                sampling_strategy='not majority',
                random_state=42
            )
            pipeline = make_pipeline(over, under)
            X_res, y_res = pipeline.fit_resample(X_train, y_train)
        else:
            print("Using standard SMOTE oversampling")
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)

        print("\nResampled class distribution:")
        print(Counter(self.le.inverse_transform(y_res)))
        return X_res, y_res

    def train_models(self, X_train, y_train):
        """Train all models with stratified CV."""
        min_samples = min(Counter(y_train).values())
        n_splits = min(5, min_samples)

        print(f"\n=== Training Models (using {n_splits}-fold CV) ===")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring='f1_weighted', n_jobs=-1
            )
            print(f"Cross-validation F1 scores: {cv_scores}")
            print(f"Mean CV F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

            model.fit(X_train, y_train)
            self.results[name] = {
                'model': model,
                'cv_scores': cv_scores.tolist(),
                'mean_cv_f1': float(np.mean(cv_scores))
            }

    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and select the best one."""
        print("\n=== Model Evaluation ===")

        for name, result in self.results.items():
            model = result['model']
            y_pred = model.predict(X_test)

            print(f"\n--- {name} Performance ---")
            print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            print(f"Test F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.3f}")

            # Get full classification report as dict
            report = classification_report(
                y_test, y_pred,
                target_names=self.le.classes_,
                output_dict=True,
                zero_division=0
            )

            # Store the full report
            result['classification_report'] = report

            # Print detailed metrics for each class
            print("\nDetailed Metrics per Class:")
            metrics_df = pd.DataFrame(report).transpose()
            print(metrics_df.iloc[:-3, :])  # Exclude averages

            # Print averages
            print("\nAverages:")
            print(metrics_df.iloc[-3:, :])

            # Store test metrics
            result['test_accuracy'] = accuracy_score(y_test, y_pred)
            result['test_f1'] = f1_score(y_test, y_pred, average='weighted')

            # Generate plots
            self._plot_confusion_matrix(y_test, y_pred, name)
            if hasattr(model, 'feature_importances_'):
                self._plot_feature_importance(model, X_test.columns, name)

        # Select best model
        self.best_model_name = max(self.results.items(),
                                 key=lambda x: x[1]['test_f1'])[0]
        self.best_model = self.results[self.best_model_name]['model']

        print("\n=== Best Model Selection ===")
        print(f"Selected {self.best_model_name} with:")
        print(f"- Test Accuracy: {self.results[self.best_model_name]['test_accuracy']:.3f}")
        print(f"- Test F1 Score: {self.results[self.best_model_name]['test_f1']:.3f}")

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Generate and display normalized confusion matrix."""
        plt.figure(figsize=(18, 12))
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.le.classes_,
                    yticklabels=self.le.classes_)
        plt.title(f'{model_name} - Confusion Matrix', pad=20)
        plt.xlabel('Predicted', labelpad=10)
        plt.ylabel('Actual', labelpad=10)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        os.makedirs('results', exist_ok=True)
        filename = f'results/confusion_matrix_{model_name}_{self.timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved confusion matrix to {filename}")
        plt.close()

    def _plot_feature_importance(self, model, feature_names, model_name, threshold=0.01):
        """Generate and display feature importance plot."""
        importances = model.feature_importances_
        indices = np.where(importances > threshold)[0]

        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name} - Feature Importance', pad=20)
        bars = plt.barh(range(len(indices)), importances[indices], color='steelblue')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance', labelpad=10)

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}',
                     va='center')

        plt.tight_layout()

        # Save plot
        filename = f'results/feature_importance_{model_name}_{self.timestamp}.png'
        plt.savefig(filename, dpi=300)
        print(f"Saved feature importance to {filename}")
        plt.close()

    def save_artifacts(self):
        """Save all models and results."""
        os.makedirs('results', exist_ok=True)

        # Save models
        for name, result in self.results.items():
            joblib.dump(result['model'], f'results/{name}_model.joblib')

        # Save production artifacts
        joblib.dump(self.best_model, 'results/production_model.joblib')
        joblib.dump(self.le, 'results/label_encoder.joblib')
        joblib.dump(self.feature_columns, 'results/feature_columns.joblib')

        # Save report
        report = {
            'timestamp': self.timestamp,
            'best_model': self.best_model_name,
            'dropped_columns': self.dropped_columns,
            'results': {
                name: {
                    'cv_scores': result['cv_scores'],
                    'mean_cv_f1': result['mean_cv_f1'],
                    'test_accuracy': result['test_accuracy'],
                    'test_f1': result['test_f1'],
                    'classification_report': result['classification_report']
                }
                for name, result in self.results.items()
            },
            'classes': self.le.classes_.tolist()
        }

        with open(f'results/training_report_{self.timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("\n=== Saved Artifacts ===")
        print(f"- Models: results/*_model.joblib")
        print(f"- Production model: results/production_model.joblib")
        print(f"- Label encoder: results/label_encoder.joblib")
        print(f"- Feature columns: results/feature_columns.joblib")
        print(f"- Training report: results/training_report_{self.timestamp}.json")
        if self.dropped_columns:
            print(f"- Dropped columns: {self.dropped_columns}")


if __name__ == "__main__":
    trainer = ModelTrainer()

    print("=== Data Loading ===")
    X_train, X_test, y_train, y_test = trainer.load_data()

    print("\n=== Preprocessing ===")
    try:
        X_train, y_train = trainer.handle_imbalance(X_train, y_train)
    except Exception as e:
        print(f"Warning: Could not apply SMOTE due to: {str(e)}")
        print("Proceeding without resampling...")

    print("\n=== Model Training ===")
    trainer.train_models(X_train, y_train)

    print("\n=== Model Evaluation ===")
    trainer.evaluate_models(X_test, y_test)

    print("\n=== Saving Results ===")
    trainer.save_artifacts()

    print("\n=== Training Complete ===")
    print(f"Best model ({trainer.best_model_name}) is ready for production use!")