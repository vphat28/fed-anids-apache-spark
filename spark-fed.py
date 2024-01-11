# Import the necessary modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pyspark.sql import SparkSession
import numpy as np
# Import the Elephas library for distributed Keras on Spark
from elephas.spark_model import SparkModel

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
  layers.Dense(256, activation="relu", input_shape=(784,)),
  layers.Dense(128, activation="relu"),
  layers.Dense(10, activation="softmax")
])

# Compile the model with loss, optimizer and metrics
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Create a Spark model from the Keras model
spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')

# Train the model on the train set using Spark
spark_model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2, verbose=0, workers=10)

# Evaluate the model on the test set using Spark
score = spark_model.evaluate(x_test, y_test, batch_size=32, verbose=0, workers=10)
print(f"The accuracy is {score[1]:.4f}")

# Stop the Spark session
spark.stop()
