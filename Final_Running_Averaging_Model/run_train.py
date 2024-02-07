# run_train.py
"""
SML Project by 
1) Ayush Manojkumar Lodha
2) Sameer Hussain
3) Aditya Gaitonde
4) Ravi Teja Seera
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('train.csv')

# Create a mapping from metaphorID to the actual metaphor words
metaphor_mapping = {
    0: 'road', 1: 'candle', 2: 'light', 3: 'spice', 4: 'ride', 5: 'train', 6: 'boat'
}
df['metaphor_word'] = df['metaphorID'].map(metaphor_mapping)

# Convert the 'label_boolean' column to integers
df['label_boolean'] = df['label_boolean'].astype(int)

# Feature extraction using TF-IDF for the text
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(df['text']).toarray()

# One-hot encode the 'metaphor_word' column
onehot_encoder = OneHotEncoder(sparse=False)
X_train_metaphor = onehot_encoder.fit_transform(df[['metaphor_word']])

# Combine TF-IDF features with one-hot encoded metaphor_word features
X_train = np.hstack((X_train_tfidf, X_train_metaphor))
y_train = df['label_boolean']

# Handle class imbalance
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define models
models = [
    RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    AdaBoostClassifier(learning_rate=0.01, n_estimators=200, random_state=42),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=10, n_estimators=200, random_state=42),
    ExtraTreesClassifier(n_estimators=100, random_state=42),
    XGBClassifier(learning_rate=0.01, max_depth=6, n_estimators=300, random_state=42, use_label_encoder=False, eval_metric='logloss')
]

# Train and save each model
for i, model in enumerate(models):
    model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(model, f'models/model_{i}.pkl')

# Save the TF-IDF vectorizer and OneHotEncoder
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(onehot_encoder, 'onehot_encoder.pkl')

print("Models, TF-IDF vectorizer, and OneHotEncoder saved.")
