import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from pyspark.sql import SparkSession
import numpy as np
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
import pyspark.sql.functions as F

# Initialize SparkSession
spark = SparkSession.builder.master("local[10]").appName("HeartDiseaseSpark").getOrCreate()

# Load the Heart Disease dataset from CSV
data = spark.read.format("csv").option("header", "true").load("/app/ids.csv")

# Trim column names and values
trimmed_column_names = [F.trim(F.col(col)).alias(col.strip()) for col in data.columns]
data = data.select(trimmed_column_names)
trimmed_data = data.select([F.trim(F.col(col)).alias(col) for col in data.columns])

# Replace null or NaN values in the 'Label' column
trimmed_data = trimmed_data.withColumn(
    "Label",
    F.when(F.col("Label").isNull() | F.isnan(F.col("Label")), 0).otherwise(F.col("Label"))
)

# Convert 'BENIGN' to 1 and other values to 0 in the 'Label' column
trimmed_data = trimmed_data.withColumn(
    "Label",
    F.when(F.col("Label") == "BENIGN", 1).otherwise(0)
)

# Cast all columns to double
for col_name in trimmed_data.columns:
    trimmed_data = trimmed_data.withColumn(col_name, F.col(col_name).cast("double"))

# Replace NaN and Infinity values in feature columns
feature_columns = [col for col in trimmed_data.columns if col != "Label"]
for col_name in feature_columns:
    trimmed_data = trimmed_data.withColumn(
        col_name,
        F.when(F.isnan(F.col(col_name)) | F.col(col_name).isNull(), 0).otherwise(F.col(col_name))
    )
    trimmed_data = trimmed_data.withColumn(
        col_name,
        F.when(F.col(col_name) == float("inf"), float("1e10")).otherwise(F.col(col_name))
    )
    trimmed_data = trimmed_data.withColumn(
        col_name,
        F.when(F.col(col_name) == -float("inf"), -float("1e10")).otherwise(F.col(col_name))
    )

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(trimmed_data)

# Normalize the feature vector using PySpark's MinMaxScaler
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)

# Convert to Pandas DataFrame and then to NumPy array
pandas_df = scaled_data.select("scaledFeatures", "Label").toPandas()
features_array = np.array(pandas_df["scaledFeatures"].tolist())
labels_array = np.array(pandas_df["Label"].tolist())

# Define the autoencoder model
encoding_dim = 32
input_dim = features_array.shape[1]

input_img = Input(shape=(input_dim,))
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-2]
decoder_layer2 = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(features_array, features_array, epochs=50, batch_size=256, shuffle=True)

# Encode and decode the features
encoded_imgs = encoder.predict(features_array)
decoded_imgs = decoder.predict(encoded_imgs)

# Calculate mean squared error as a reconstruction error metric
reconstruction_error = np.mean(np.power(features_array - decoded_imgs, 2), axis=1)

# Define a threshold for reconstruction error above which a sample is considered an anomaly
threshold = np.percentile(reconstruction_error, 95)

# Predict labels based on reconstruction error
predicted_labels = (reconstruction_error > threshold).astype(int)

# Calculate accuracy
accuracy = np.mean(predicted_labels == labels_array)
print("\033[32mMean Squared Error:\033[0m", np.mean(reconstruction_error))  # Green
print("\033[32mAccuracy:\033[0m", accuracy)  # Green

# Stop Spark
spark.stop()
