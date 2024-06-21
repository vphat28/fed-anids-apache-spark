from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from elephas.utils.rdd_utils import to_simple_rdd
from elephas.spark_model import SparkModel
from keras.models import Sequential, Model
from keras.layers import Input, Dense

import pyspark.sql.functions as F

# Step 1: Initialize SparkContext and SparkSession
conf = SparkConf().setAppName('AutoencoderTraining').setMaster('local[*]')
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Step 2: Load and preprocess your dataset (replace with your actual preprocessing steps)
data = spark.read.format("csv").option("header", "true").load("/app/ids.csv")
trimmed_column_names = [F.trim(F.col(col)).alias(col.strip()) for col in data.columns]
data = data.select(trimmed_column_names)

# Trim spaces from all column values
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
data = trimmed_data
# Step 3: Convert data to RDD for Elephas
rdd_data = to_simple_rdd(spark, data)

# Step 4: Define the Autoencoder model using Keras
input_dim = len(data.columns)  # Number of features
encoding_dim = 32

input_layer = Input(shape=(input_dim,))
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_layer, decoded)

# Step 5: Wrap the model with SparkModel
spark_model = SparkModel(autoencoder, frequency='epoch', mode='asynchronous')

# Step 6: Train the autoencoder model using SparkModel
spark_model.train(rdd_data, nb_epoch=10, batch_size=256, verbose=1)

# Step 7: Stop SparkSession
spark.stop()
