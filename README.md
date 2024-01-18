- Build image in docker-files/fed-learning `cd docker-files/fed-learning/` `docker build . -t fed-spark-node`
- Start docker cluster `docker-compose up`
- Submit job `docker-compose exec spark-master  spark-submit --master spark://spark-master:7077 /app/spark-fed-heart.
py`
