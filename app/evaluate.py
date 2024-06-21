import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_squared_error, accuracy_score

# Step 1: Load the trained autoencoder model
autoencoder = load_model('trained_autoencoder.h5')

# Step 2: Load and preprocess the Heart Disease dataset (heart.csv)
heart_data = pd.read_csv('heart.csv')

# Assuming 'target' is the column indicating presence (1) or absence (0) of heart disease
# Drop the 'target' column for unsupervised learning (autoencoder)
features = heart_data.drop('target', axis=1)

# Check the number of features
print(f"Number of features: {len(features.columns)}")



# Convert features to numpy array
input_data = np.array(heart_data)

# Step 3: Predict with the autoencoder model
reconstructed_data = autoencoder.predict(input_data)

# Step 4: Calculate reconstruction error (MSE)
mse = mean_squared_error(input_data, reconstructed_data)

# Step 5: Classify records based on MSE threshold
threshold = 150  # Set your threshold here
heart_disease_predictions = (mse > threshold).astype(int)

# Step 6: Calculate accuracy
actual_labels = np.array(heart_data['target'])  # Assuming 'target' column contains actual labels (1 for heart disease, 0 for no heart disease)

accuracy = accuracy_score(actual_labels, heart_disease_predictions)

# Step 7: Print accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')
