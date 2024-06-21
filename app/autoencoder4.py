from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when, isnan
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from keras.models import Model, model_from_json
from keras.layers import Input, Dense
import numpy as np
import json

# Step 1: Initialize SparkContext and SparkSession
spark_context = SparkContext.getOrCreate()
spark = SparkSession.builder.master("local[10]").appName("AnomalyDetection").getOrCreate()

# Load and preprocess your dataset
data = spark.read.format("csv").option("header", "true").load("/app/heart.csv")

# Rename columns to remove leading and trailing spaces
renamed_columns = [trim(col(c)).alias(c.strip()) for c in data.columns]
data = data.select(renamed_columns)

# Trim spaces from all column values and cast to double
for column in data.columns:
    data = data.withColumn(column, trim(col(column)).cast("double"))

# Replace NaN and Infinity values in feature columns
feature_columns = data.columns  # All columns since we are not using labels
for column in feature_columns:
    data = data.withColumn(
        column,
        when(isnan(col(column)) | col(column).isNull(), 0).otherwise(col(column))
    )
    data = data.withColumn(
        column,
        when(col(column) == float("inf"), float("1e10")).otherwise(col(column))
    )
    data = data.withColumn(
        column,
        when(col(column) == -float("inf"), -float("1e10")).otherwise(col(column))
    )

# Collect features into numpy array
features_array = np.array(data.select(feature_columns).collect())

# Convert to RDD format expected by Elephas
rdd_data = to_simple_rdd(spark_context, features_array, features_array)  # Use features_array as both features and labels

# Define the Autoencoder model using Keras
input_dim = len(feature_columns)  # Number of features
encoding_dim = 32

def create_autoencoder():
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(256, activation='relu')(input_layer)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

# Create and compile the model
autoencoder = create_autoencoder()

# Serialize the model to JSON and weights to HDF5
model_json = autoencoder.to_json()
with open("/app/autoencoder_model.json", "w") as json_file:
    json_file.write(model_json)
autoencoder.save_weights("/app/autoencoder_model.h5")

# Function to load and compile the model within each worker
def load_and_compile_model():
    with open("/app/autoencoder_model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("/app/autoencoder_model.h5")
    loaded_model.compile(optimizer='adam', loss='mean_squared_error')
    return loaded_model

# Load and compile the model within each Spark worker
def train_model(iterator):
    model = load_and_compile_model()
    for features in iterator:
        features = np.array(features)
        model.fit(features, features, epochs=1, batch_size=256, verbose=1)
    yield model.get_weights()

# Perform the training
weights = rdd_data.mapPartitions(train_model).collect()

# Update the model with the trained weights
model = create_autoencoder()
average_weights = np.mean(weights, axis=0)
model.set_weights(average_weights)

# Evaluate the model on the training data (compute reconstruction error)
reconstruction_error = model.evaluate(features_array, features_array, verbose=1)

print(f"Final Model Reconstruction Error: {reconstruction_error}")
# Save the trained model
model.save("/app/trained_autoencoder.h5")

# Stop SparkSession
spark.stop()
