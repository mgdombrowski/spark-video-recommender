#!/bin/bash

SPARK_DIRECTORY="/Users/mattd/Downloads/spark-1.2.1-bin-hadoop2.4"
LOCAL_REPO="/Users/mattd/Documents/Development/git/mgdombrowski/video-recommender"


${SPARK_DIRECTORY}"/bin/spark-submit" --master local[4] ${LOCAL_REPO}"/recommender.py"
