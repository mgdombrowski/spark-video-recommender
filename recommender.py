from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS
import yaml
import requests
import json

# TODO: get API keys from mySQL/ActiveRecord
# load settings from config file
doc = open('config.yaml', 'r')
config = yaml.load(doc)

# profiles API key/secret for org
api_key = config['key']
api_secret = config['secret']

api_root = "https://api.localytics.com/profile/v1/profiles/"


def profile_add(customer_id, product):
    # add recommended video to set of user videos
    profile_data = json.dumps(
        {
            'changes': [
                {
                    'op': 'set-add',
                    'attr': 'Recommended Videos',
                    'value': [str(product)]
                }
            ]
        })
    r = requests.patch(api_root + customer_id, profile_data, auth=(api_key, api_secret),
                       headers={'Content-Type': 'application/json'})
    print r
    print r.json()

# for local running
sc = SparkContext("local")
sqlContext = SQLContext(sc)

# TODO: research default cluster partitioning vs other schemes
#logs = sqlContext.jsonFile("s3n://exports.localytics.n-q/nhl/2015/*/*/4fde3bff810b8ea29af5bf9-c6d5bd8e-2491-11e4-a2bd-009c5fda0a25/*.gz")
logs = sqlContext.jsonFile("s3n://exports.localytics.n-q/nhl/2015/01/01/4fde3bff810b8ea29af5bf9-c6d5bd8e-2491-11e4-a2bd-009c5fda0a25/00.log.gz")

logs.registerTempTable("logs")

# TODO: for temporal effects/online usage--replace old recommendations using incremental collaborative filtering

# TODO: abstract 'Video Watched' to any event with feedback from dashboard (mysql table?)
query = sqlContext.sql(
    "select customer_ids.customer_id,custom from logs \
    where name = 'Video Watched' and customer_ids.customer_id is not null and custom is not null")

# remove video URLs without PID
clean_query = query.filter(lambda x: 'pid=' in x.custom.asDict()['Raw Content URL'])
clean_query.persist(StorageLevel.MEMORY_AND_DISK)

data = clean_query.map(
    lambda p: p.customer_id + ',' + p.custom.asDict()['Raw Content URL'].split('pid=', 1)[1][0:6].replace('&', ''))
clean_query.unpersist()
data.persist(StorageLevel.MEMORY_AND_DISK)

# TODO: for millions of users, find better hashing algorithm to avoid collisions
# Alternating Least Squares (ALS) algorithm requires integer values, so hash string to int
def hash_8(astring):
    return abs(hash(astring)) % (10 ** 8)

# build hash table so we can recover customer ID later
# TODO: fix
hash_table = {}
data.foreach(lambda i: hash_table.update({hash_8(i.split(',')[0]): str(i.split(',')[0])}))

# works but is slow
# for i in data.collect():
#     hash_table[hash_8(i.split(',')[0])] = str(i.split(',')[0])

# TODO: rank preference based on values of attribute, e.g.:
# 1 - Viewed 0-25%
# 2 - Viewed 25-50%
# 3 - Viewed 50-75%
# 4 - Viewed 75-100%
# or based on different events
# 1 - Viewed
#    2 - Liked
#    3 - Shared
#    4 - Added to Cart
#    5 - Purchased

# get results in format for Alternating Least Squares matrix factorization (user: Int, product: Int, rating: Double)
ratings = data.map(lambda l: l.split(',')).map(lambda x: Rating(hash_8(x[0]), x[1], 1.0))
data.unpersist()
ratings.persist(StorageLevel.MEMORY_AND_DISK)

# TODO: kfold cross validation
# split into training, validation, and test sets
training = ratings.sample(False, 0.33)
training.persist(StorageLevel.MEMORY_AND_DISK)

validation = ratings.sample(False, 0.33)

test = ratings.sample(False, 0.33)
test.persist(StorageLevel.MEMORY_AND_DISK)

rank = 10
numIterations = 20
alpha = 0.01
#autocompute
numBlocks = -1

# for explicit ratings if the app has them
# model = ALS.train(training, rank, numIterations)

# Build the recommendation model using Alternating Least Squares based on implicit ratings
# TODO: tune model by looping through some variations of model parameters
# lambda, alpha, rank, numIterations
# TODO: warning...Product factor does not have a partitioner. Prediction on individual records could be slow.
model = ALS.trainImplicit(training, rank, numIterations, alpha=alpha, blocks=numBlocks)
training.unpersist()

testdata = test.map(lambda p: (int(p[0]), int(p[1])))
test.unpersist()
testdata.persist()

# TODO: change to full data set once model is tuned
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
predictions.persist()

ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
ratings.unpersist()
ratesAndPreds.persist(StorageLevel.MEMORY_AND_DISK)

# TODO: Recommendation explanation--for each user, find past action with highest contribution to recommendation
# this is the "recommended based on your viewing of X" explanation

#evaluate the model using root-mean-square error
MSE = ratesAndPreds.distinct().map(lambda r: (r[1][0] - r[1][1]) ** 2).reduce(
    lambda x, y: x + y) / ratesAndPreds.distinct().count()
ratesAndPreds.unpersist()

print("Mean Squared Error = " + str(MSE))

# unhash customer ID, sort, filter on strong recommendations only
recommendations = predictions.distinct().map(lambda y: ((hash_table[y[0][0]], y[0][1]), y[1])).sortByKey().filter(
    lambda z: z[1] > 0.7)
recommendations.persist(StorageLevel.MEMORY_AND_DISK)
predictions.unpersist()

#upload to Localytics Profiles DB
recommendations.foreach(lambda a: profile_add(a[0][0], a[0][1]))

# TODO: add logging

# TODO: feedback mechanism for recommendations, e.g. "Was this suggestion relevant?", A/B testing hooks, etc
