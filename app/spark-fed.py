import keras
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from pyspark.sql import SparkSession
import numpy as np
from elephas.spark_model import SparkModel
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from elephas.utils.rdd_utils import to_simple_rdd
import pyspark.ml.classification as cl
import pyspark.sql.functions as F

# Initialize SparkContext and SparkSession
spark_context = SparkContext.getOrCreate()
spark = SparkSession.builder.master("local[10]").appName("HeartDiseaseSpark").getOrCreate()

# Load the Heart Disease dataset from CSV
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

# Create logistic regression model
lr = cl.LogisticRegression(maxIter=100, labelCol="Label", featuresCol="features")  # Adjust hyperparameters as needed

# Fit the model
print("We are about to take off")
model = lr.fit(assembled_data)

print("Training is done")

# Evaluate the model
predictions = model.transform(assembled_data)

# Calculate accuracy
correct_predictions = predictions.filter(predictions.Label == predictions.prediction).count()
total_data = assembled_data.count()
accuracy = correct_predictions / float(total_data)

print("\033[32mAccuracy:\033[0m", accuracy)  # Green

spark.stop()
