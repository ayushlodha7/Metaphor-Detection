# run_test.py
"""
SML Project by 
1) Ayush Manojkumar Lodha
2) Sameer Hussain
3) Aditya Gaitonde
4) Ravi Teja Seera
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Load the test dataset
test_df = pd.read_csv('train.csv')

metaphor_mapping = {
    0: 'road', 1: 'candle', 2: 'light', 3: 'spice', 4: 'ride', 5: 'train', 6: 'boat'
}
test_df['metaphor_word'] = test_df['metaphorID'].map(metaphor_mapping)

# Load the saved TF-IDF vectorizer and one-hot encoder
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
onehot_encoder = joblib.load('onehot_encoder.pkl')

# Apply the TF-IDF vectorizer and one-hot encoder to the test data
X_test_tfidf = tfidf_vectorizer.transform(test_df['text']).toarray()
X_test_metaphor = onehot_encoder.transform(test_df[['metaphor_word']])
X_test = np.hstack((X_test_tfidf, X_test_metaphor))

# Load and predict with each model
prob_sum = np.zeros((len(X_test), 2))  # Adjust based on your classification needs
num_models = 5  # Change this to the number of models you have
for i in range(num_models):
    model = joblib.load(f'models/model_{i}.pkl')
    probs = model.predict_proba(X_test)
    prob_sum += probs

# Compute the average probabilities
avg_prob = prob_sum / num_models

# Determine final predictions based on the highest probability
final_predictions = np.argmax(avg_prob, axis=1)

# Compute and display the performance metrics
print("Ensemble Model Performance on Test Data:")
print(classification_report(test_df['label_boolean'], final_predictions))
print(f"Accuracy: {accuracy_score(test_df['label_boolean'], final_predictions)}\n")

# Save the predictions to a CSV file
test_df['Predictions'] = final_predictions
test_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv.")
