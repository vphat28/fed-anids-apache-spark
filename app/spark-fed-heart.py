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

spark_context = SparkContext.getOrCreate()
spark = SparkSession.builder.master("local[10]").appName("HeartDiseaseSpark").getOrCreate()
print("hello xavi")
# Load the Heart Disease dataset from CSV
data = spark.read.format("csv").option("header", "true").load("/app/heart.csv")
# Get a list of all column names
column_names = data.columns

# Iterate through the columns and cast each to integer
for col_name in column_names:
    data = data.withColumn(col_name, F.col(col_name).cast("double"))
# Select features and target variable
data = data.select("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target")

# Convert features to NumPy array
features = data.select("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal").rdd.map(lambda row: row[0]).collect()
target = data.select("target").rdd.map(lambda row: row[0]).collect()

# Normalize features (if necessary)
# ...
# Assemble features into a single vector
assembler = VectorAssembler(inputCols=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"], outputCol="features")
assembled_data = assembler.transform(data)

# Create logistic regression model
lr = cl.LogisticRegression(maxIter=100, labelCol="target")  # Adjust hyperparameters as needed

# Fit the model
print("We are about to take off")
model = lr.fit(assembled_data)

print("Training is done")
# Evaluate the model
predictions = model.transform(assembled_data)
accuracy = predictions.filter(predictions.prediction == predictions.target).count() / float(assembled_data.count())
print("\033[32mAccuracy.\033[0m", accuracy)  # Green
spark.stop()
