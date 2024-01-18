# Import the necessary modules
import keras
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from pyspark.sql import SparkSession
import numpy as np
# Import the Elephas library for distributed Keras on Spark
from elephas.spark_model import SparkModel
from pyspark import SparkContext

spark_context = SparkContext.getOrCreate()

# Create a Spark session with 10 nodes
spark = SparkSession.builder.master("local[10]").appName("Spark MNIST Example").getOrCreate()

# Load the MNIST data as a NumPy array
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the images to the range of [0, 1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Flatten the images to 1D vectors
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Define the model architecture
model = keras.Sequential([
  Dense(256, activation="relu", input_shape=(784,)),
  Dense(128, activation="relu"),
  Dense(10, activation="softmax")
])

# Compile the model with loss, optimizer and metrics
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Create a Spark model from the Keras model
spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')

# Convert numpy to spark dataframe
x_train_df = spark_context.parallelize(x_train, 10)

# Train the model on the train set using Spark
spark_model.fit(x_train_df, validation_ratio=0.2, batch_size=32, epochs=100, verbose=0, workers=10)

# Evaluate the model on the test set using Spark
score = spark_model.evaluate(x_test, y_test, batch_size=32, verbose=0, workers=10)
print(f"The accuracy is {score[1]:.4f}")

# Stop the Spark session
spark.stop()
