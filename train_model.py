import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


def load_data():
    """Load and preprocess data with proper encoding."""
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Encode labels consistently
    le = LabelEncoder()
    le.fit(pd.concat([train_df['prognosis'], test_df['prognosis']]))

    X_train = train_df.drop('prognosis', axis=1)
    y_train = le.transform(train_df['prognosis'])

    X_test = test_df.drop('prognosis', axis=1)
    y_test = le.transform(test_df['prognosis'])

    print("\nClass distribution in training set:")
    print(Counter(le.inverse_transform(y_train)))

    print("\nClass distribution in test set:")
    print(Counter(le.inverse_transform(y_test)))

    return X_train, X_test, y_train, y_test, le


def handle_imbalance(X_train, y_train):
    """Handle class imbalance with adaptive resampling."""
    class_counts = Counter(y_train)
    min_samples = min(class_counts.values())

    # For classes with very few samples, we'll use different strategies
    if min_samples < 5:
        print("\nWarning: Some classes have very few samples (<5). Using combined resampling strategy.")

        # Create a pipeline that first oversamples minority classes then undersamples majority
        over = SMOTE(sampling_strategy='not minority', k_neighbors=min(5, min_samples - 1), random_state=42)
        under = RandomUnderSampler(sampling_strategy='not majority', random_state=42)
        pipeline = make_pipeline(over, under)

        X_res, y_res = pipeline.fit_resample(X_train, y_train)
    else:
        # Standard SMOTE when we have enough samples
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

    print("\nAfter resampling class distribution:")
    print(Counter(y_res))
    return X_res, y_res


def train_model(X_train, y_train):
    """Train optimized Random Forest with stratified CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        random_state=42,
        max_depth=12,
        min_samples_split=3,
        max_features='log2',
        n_jobs=-1
    )

    print("\nRunning cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=cv, scoring='f1_weighted', n_jobs=-1)
    print(f"CV F1 scores: {cv_scores}")
    print(f"Mean CV F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, le):
    """Enhanced evaluation with better visualizations."""
    y_pred = model.predict(X_test)
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

    plt.figure(figsize=(18, 16))
    cm = confusion_matrix(y_test_labels, y_pred_labels,
                          labels=le.classes_, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Normalized Confusion Matrix', pad=20)
    plt.xlabel('Predicted', labelpad=10)
    plt.ylabel('Actual', labelpad=10)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_feature_importance(model, X_test.columns)


def plot_feature_importance(model, feature_names, threshold=0.01):
    """Improved feature importance visualization."""
    importances = model.feature_importances_
    indices = np.where(importances > threshold)[0]

    plt.figure(figsize=(12, 8))
    plt.title(f'Important Symptoms (Threshold > {threshold})', pad=20)
    bars = plt.barh(range(len(indices)), importances[indices], color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance', labelpad=10)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}',
                 va='center')

    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300)
    plt.close()


def save_artifacts(model, le, feature_columns):
    """Save all necessary artifacts for deployment."""
    os.makedirs('results', exist_ok=True)
    joblib.dump(model, 'results/disease_predictor.joblib')
    joblib.dump(le, 'results/label_encoder.joblib')
    joblib.dump(list(feature_columns), 'results/feature_columns.joblib')
    print("\nSaved artifacts:")
    print("- Trained model (disease_predictor.joblib)")
    print("- Label encoder (label_encoder.joblib)")
    print("- Feature columns (feature_columns.joblib)")


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, le = load_data()

    print("\nHandling class imbalance...")
    try:
        X_train, y_train = handle_imbalance(X_train, y_train)
    except Exception as e:
        print(f"\nWarning: Could not apply SMOTE due to: {str(e)}")
        print("Proceeding without resampling...")

    print("\nTraining model...")
    model = train_model(X_train, y_train)

    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test, le)

    # Save all artifacts
    save_artifacts(model, le, X_train.columns)

    print("\nTraining complete!")