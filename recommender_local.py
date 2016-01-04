from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.mllib.recommendation import Rating
from pyspark.mllib.recommendation import ALS

# for local running
sc = SparkContext("local")
sqlContext = SQLContext(sc)

# TODO: research default cluster partitioning vs other schemes
# TODO: change to s3 source; scale
logs = sqlContext.jsonFile("/tmp/files/00.log")
# logs = sqlContext.jsonFile(
# "s3n://exports.localytics.n-q/nhl/2015/01/25/4fde3bff810b8ea29af5bf9-c6d5bd8e-2491-11e4-a2bd-009c5fda0a25/*.gz")
logs.persist()

logs.registerTempTable("logs")

# TODO: for temporal effects/online usage--replace old recommendations using incremental collaborative filtering

query = sqlContext.sql(
    "select customer_ids.customer_id,custom from logs \
    where name = 'Video Watched' and customer_ids.customer_id is not null and custom is not null")
query.persist()

data = query.map(lambda p: p.customer_id + ',' + p.custom.asDict()['Raw Content URL'])
data.persist()


# Alternating Least Squares (ALS) algorithm requires integer values, so hash string to int
def hash_8(astring):
    return abs(hash(astring)) % (10 ** 8)

# TODO: save to text file
hash_table = {}
for i in data.collect():
    for p in i.split(','):
        hash_table[str(hash_8(p))] = str(p)

# get results in format for Alternating Least Squares matrix factorization (user: Int, product: Int, rating: Double)
ratings = data.map(lambda l: l.split(',')).map(lambda x: Rating(hash_8(x[0]), hash_8(x[1]), 1.0))
ratings.persist()

# TODO: kfold cross validation
# split into training, validation, and test sets
training = ratings.sample(False, 0.33)
training.persist()
validation = ratings.sample(False, 0.33)
test = ratings.sample(False, 0.33)

rank = 10
numIterations = 20

# for explicit ratings if the app has them
# model = ALS.train(training, rank, numIterations)

# Build the recommendation model using Alternating Least Squares based on implicit ratings
# TODO: tune model by looping through some variations of model parameters
# lambda, alpha, rank, numIterations
model = ALS.trainImplicit(training, rank, numIterations, alpha=0.01)

testdata = test.map(lambda p: (int(p[0]), int(p[1])))
testdata.persist()
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
predictions.persist()
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
ratesAndPreds.persist()

# TODO: Recommendation explanation--for each user, find past action with highest contribution to recommendation
# this is the "recommended based on your viewing of X" explanation

# TODO: refactor/find better RMSE calculation
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).reduce(lambda x, y: x + y) / ratesAndPreds.count()
print("Mean Squared Error = " + str(MSE))

# strong SKU recommendations for users:
output = predictions.filter(lambda line: line[1] > .5)
output.persist()

for i in output.collect():
    print i

# TODO: write output to dynamo/postgres/memcached/profilesdb for storage/querying from client
predictions.saveAsTextFile("/tmp/spark_output")

# TODO: feedback mechanism for recommendations, e.g. "Was this suggestion relevant?", A/B testing, etc
