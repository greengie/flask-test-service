from os.path import expanduser, join, abspath

from pyspark.sql import SparkSession
from pyspark import HiveContext, SQLContext
from pyspark.sql import Row
import json

# warehouse_location points to the default location for managed databases and tables
# warehouse_location = abspath('spark-warehouse')

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL Hive integration example") \
    .enableHiveSupport() \
    .getOrCreate()

sc = spark.sparkContext
hiveContext = HiveContext(sc)

def get_feature(inputId):
    # schema = {'src': ['key, value']}
    print ('get feature')
    queryString = "SELECT * FROM BankTest WHERE id={}".format(inputId)
    results = hiveContext.sql(queryString)
    results.show()
    # print (type(results.toJSON().collect()))
    # results.toJSON().collect()[0]results.toJSON().collect()[0]
    return json.loads(results.toJSON().collect()[0])

# if __name__ == '__main__':
#     results = get_feature(2)
#     print (results)
#     print (type(results))