#!/bin/bash

AWS_KEY="mdombrowski-hackathon"
KEY_FILE="/Users/mattd/Downloads/mdombrowski-hackathon.pem"



/Users/mattd/Downloads/spark-1.2.0-bin-hadoop2.4/ec2/spark-ec2 -k ${AWS_KEY} -i ${KEY_FILE} -s 15 --copy-aws-credentials -t m3.large launch video-recommender
HOST=$(/Users/mattd/Downloads/spark-1.2.0-bin-hadoop2.4/ec2/spark-ec2 -k ${AWS_KEY} -i ${KEY_FILE} get-master video-recommender|tail -1)
/Users/mattd/Downloads/spark-1.2.0-bin-hadoop2.4/bin/spark-submit --master $(echo 'spark://'$HOST':7077') /Users/mattd/Documents/Development/git/mgdombrowski/video-recommender/recommender.py
/Users/mattd/Downloads/spark-1.2.0-bin-hadoop2.4/ec2/spark-ec2 destroy video-recommender