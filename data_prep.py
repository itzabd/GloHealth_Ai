import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load dataset with first row as header
df = pd.read_csv('data/symbipredict_2022.csv', header=0)

# Step 1: Transpose the dataframe (swap rows and columns)
df = df.T  # Now diseases are rows, symptoms are columns

# Step 2: Set the first row (original column names) as column names
df.columns = df.iloc[0]
df = df[1:]  # Remove the first row (which was symptom names)

# Step 3: Reset index to make diseases a column
df = df.reset_index().rename(columns={'index': 'disease'})

# Step 4: Melt the dataframe to create symptom-value pairs
melted_df = pd.melt(df, id_vars=['disease'],
                    var_name='symptom',
                    value_name='weight')

# Convert weight to numeric, coercing errors to NaN
melted_df['weight'] = pd.to_numeric(melted_df['weight'], errors='coerce')

# Step 5: Filter only significant symptom-weight pairs
# Now properly handling numeric comparison
filtered_df = melted_df[melted_df['weight'] > 0]

# Step 6: Create one-hot encoded symptoms for each disease
final_df = pd.pivot_table(filtered_df,
                         index='disease',
                         columns='symptom',
                         values='weight',
                         fill_value=0).reset_index()

# Step 7: Encode diseases
le = LabelEncoder()
final_df['disease_encoded'] = le.fit_transform(final_df['disease'])

# Save encoder
joblib.dump(le, 'label_encoder.pkl')

# Prepare features and target
X = final_df.drop(['disease', 'disease_encoded'], axis=1)
y = final_df['disease_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

# Save processed data
X_train.to_csv('data/X_train.csv', index=False)
X_val.to_csv('data/X_val.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_val.to_csv('data/y_val.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("Data preparation complete!")
print(f"Final feature shape: {X.shape}")
print(f"Unique diseases: {len(le.classes_)}")