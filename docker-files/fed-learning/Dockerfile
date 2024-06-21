FROM bitnami/spark:latest

USER root

# Install Python and pip (if not already present)
RUN apt-get update && apt-get install -y python3-pip

# Install TensorFlow and PySyft
RUN pip3 install tensorflow pyspark
RUN pip3 install sklearn
RUN pip3 install --default-timeout=100 elephas
RUN pip3 install pandas

# Set environment variables (optional)
ENV SPARK_HOME=/opt/bitnami/spark
ENV PYTHONPATH=/opt/bitnami/spark/python/lib:/opt/bitnami/spark/python/lib/python3.8/site-packages

# Copy your Python code and Spark configuration files (if needed)
#COPY . /app

#WORKDIR /app
