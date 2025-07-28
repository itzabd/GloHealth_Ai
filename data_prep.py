import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")


def load_and_clean_data(filepath):
    """Load and clean the raw dataset."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    symptom_cols = df.columns[:-1]
    df[symptom_cols] = df[symptom_cols].astype(int)
    return df.drop_duplicates()


def explore_data(df):
    """Perform exploratory data analysis."""
    print("\nDisease counts:\n", df['prognosis'].value_counts())
    print("\nTop 20 symptoms:\n", df.drop('prognosis', axis=1).sum().sort_values(ascending=False).head(20))


def preprocess_data(df, test_size=0.2):
    """Preprocess data with SMOTE only for classes with sufficient samples."""
    # Remove rare/common symptoms
    total = len(df)
    symptom_sums = df.drop('prognosis', axis=1).sum()
    to_drop = symptom_sums[(symptom_sums < 0.05 * total) | (symptom_sums > 0.95 * total)].index
    df_filtered = df.drop(columns=to_drop)
    print(f"\nRemoved {len(to_drop)} symptoms.")

    # Split before SMOTE to avoid data leakage
    X = df_filtered.drop('prognosis', axis=1)
    y = df_filtered['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Apply SMOTE only to classes with >=6 samples in training set
    class_counts = y_train.value_counts()
    valid_classes = class_counts[class_counts >= 6].index
    mask = y_train.isin(valid_classes)

    if len(valid_classes) > 0:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(class_counts[valid_classes]) - 1))
        X_res, y_res = smote.fit_resample(X_train[mask], y_train[mask])

        # Combine resampled and original minority classes
        X_train = pd.concat([X_res, X_train[~mask]])
        y_train = pd.concat([y_res, y_train[~mask]])
    else:
        print("Warning: No classes with sufficient samples for SMOTE")

    # Save processed data
    pd.concat([X_train, y_train], axis=1).to_csv("data/train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv("data/test.csv", index=False)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_and_clean_data("data/symbipredict_2022.csv")
    explore_data(df)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("\nFinal shapes:")
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")