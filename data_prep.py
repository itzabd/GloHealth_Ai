import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def load_and_clean_data(filepath):
    """Load and enhance data with geo-temporal features."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Add geo-temporal columns if not present (mock data for demonstration)
    if 'zip' not in df.columns:
        import random
        df['zip'] = [str(random.randint(10000, 99999)) for _ in range(len(df))]
        df['timestamp'] = pd.to_datetime(datetime.now()).strftime('%Y-%m-%d')

    # Convert symptoms to integers
    symptom_cols = [col for col in df.columns if col not in ['prognosis', 'zip', 'timestamp']]
    df[symptom_cols] = df[symptom_cols].astype(int)

    return df.drop_duplicates()


def preprocess_data(df, test_size=0.2):
    """Preprocess data while preserving geo-temporal information."""
    # Define columns to preserve
    geo_cols = ['zip', 'timestamp']
    symptom_cols = [col for col in df.columns if col not in ['prognosis'] + geo_cols]

    # Remove rare/common symptoms (only from symptom columns)
    total = len(df)
    symptom_sums = df[symptom_cols].sum()
    to_drop = symptom_sums[(symptom_sums < 0.05 * total) | (symptom_sums > 0.95 * total)].index
    df_filtered = df.drop(columns=to_drop)
    print(f"\nRemoved {len(to_drop)} symptoms.")

    # Split data (preserving geo columns)
    X = df_filtered.drop('prognosis', axis=1)
    y = df_filtered['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Apply SMOTE only to symptom features
    class_counts = y_train.value_counts()
    valid_classes = class_counts[class_counts >= 6].index
    mask = y_train.isin(valid_classes)

    if len(valid_classes) > 0:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(class_counts[valid_classes]) - 1))

        # Fit SMOTE and transform
        X_res, y_res = smote.fit_resample(
            X_train[mask].drop(geo_cols, axis=1),
            y_train[mask]
        )

        # Get the indices of resampled points
        n_samples = len(X_res) - sum(mask)  # Number of new samples
        original_indices = np.where(mask)[0]
        new_indices = np.concatenate([
            original_indices,
            np.array([original_indices[0]] * n_samples)  # Placeholder, will be replaced
        ])

        # Recombine with geo data
        X_geo_resampled = pd.concat([
            X_train[geo_cols].iloc[original_indices],
            X_train[geo_cols].iloc[[original_indices[0]] * n_samples]  # Duplicate geo data
        ], axis=0)

        X_train = pd.concat([
            pd.DataFrame(X_res, columns=X_train.drop(geo_cols, axis=1).columns),
            X_geo_resampled.reset_index(drop=True),
            X_train[~mask].drop(geo_cols, axis=1),
            X_train[geo_cols][~mask].reset_index(drop=True)
        ], axis=1)

        y_train = pd.concat([y_res, y_train[~mask]])
    else:
        print("Warning: No classes with sufficient samples for SMOTE")

    # Ensure column order consistency
    final_cols = X.columns.tolist()
    X_train = X_train[final_cols]
    X_test = X_test[final_cols]

    # Save processed data (including geo columns)
    pd.concat([X_train, y_train], axis=1).to_csv("data/train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv("data/test.csv", index=False)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_and_clean_data("data/symbipredict_2022.csv")
    print("\nOriginal data columns:", df.columns.tolist())
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\nFinal datasets:")
    print(f"Train: {X_train.shape}, Geo columns: {['zip', 'timestamp']}")
    print(f"Test: {X_test.shape}")
    print("\nSample training data:")
    print(pd.concat([X_train.head(2), y_train.head(2)], axis=1))