# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import os

# Correct file path using relative directory
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, '16P.csv')

# Read CSV and assign it to df
df = pd.read_csv(csv_path, encoding='latin1')

# Show column names
z = df.columns
print("Columns:", z)

# Head of DataFrame (optional)
print(df.head())

# Drop 'Response Id' column
df.drop('Response Id', axis=1, inplace=True)

# Save original question column names
questions = list(df.columns)

print("Questions of the test:")
for i, q in enumerate(df.columns, start=1):
    print(f'\t{i}-{q}')

# Drop duplicates
df = df.drop_duplicates()

# Separate labels
y = df["Personality"].copy()
df.drop("Personality", axis=1, inplace=True)

# Rename columns to Q1, Q2, ...
qq = ['Q'+str(i) for i in range(1, len(df.columns)+1)]
df.columns = qq

# Map short names to full questions
questions_map = {qq[i]: questions[i] for i in range(len(qq))}

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

# Train SVM model
sv = SVC(C=1, gamma=0.01, kernel='rbf')
sv.fit(x_train, y_train)

# Print accuracy
print("Train Accuracy:", sv.score(x_train, y_train))
print("Test Accuracy:", sv.score(x_test, y_test))

# Save model, encoder, and filtered question names
joblib.dump(sv, os.path.join(current_dir, 'svc_model.pkl'))
joblib.dump(le, os.path.join(current_dir, 'label_encoder.pkl'))

columns_filtered = [col for col in df.columns if 'Personality' not in col]
joblib.dump(columns_filtered, os.path.join(current_dir, 'questions.pkl'))
