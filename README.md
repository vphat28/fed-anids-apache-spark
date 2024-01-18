- Build image in docker-files/fed-learning `docker build . -t fed-spark-node`
- docker-compose exec spark-master  spark-submit --master spark://spark-master:7077 /app/spark-fed.
py