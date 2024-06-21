from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
import numpy as np
from pyspark import SparkContext
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import pyspark.sql.functions as F
# Step 1: Initialize SparkSession
spark_context = SparkContext.getOrCreate()
spark = SparkSession.builder.master("local[*]").appName("DistributedAutoencoderTraining").getOrCreate()

data = spark.read.format("csv").option("header", "true").load("/app/ids.csv")

# Rename columns to remove leading and trailing spaces
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

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_data = assembler.transform(trimmed_data)
data = assembler.transform(trimmed_data)

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(data)
scaled_data = scaler_model.transform(data)

# Convert Spark DataFrame to Pandas DataFrame and then to NumPy array for TensorFlow/Keras
pandas_df = scaled_data.select("scaledFeatures").toPandas()
features_array = np.array(pandas_df["scaledFeatures"].tolist())


# Step 4: Define and train the autoencoder using TensorFlow/Keras on each Spark worker
def train_autoencoder(partition):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model

    # Initialize autoencoder model
    input_dim = features_array.shape[1]
    input_img = Input(shape=(input_dim,))
    encoded = Dense(8, activation='relu')(input_img)  # Adjust architecture as needed
    encoded = Dense(4, activation='relu')(encoded)
    decoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the autoencoder on partition data
    partition_data = list(partition)
    partition_features = np.array([row.scaledFeatures.toArray() for row in partition_data])
    autoencoder.fit(partition_features, partition_features, epochs=10, batch_size=256, shuffle=True)

    # Optionally return trained model or metrics


# Perform distributed training using mapPartitions
trained_models = scaled_data.rdd.mapPartitions(train_autoencoder).collect()

# Step 5: Aggregate results if needed (e.g., model parameters or metrics)

# Step 6: Stop SparkSession
spark.stop()
