from pyspark import SparkConf, SparkContext, SQLContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
import sys, operator, json, string,re,pickle,ast
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.mllib.feature import Word2Vec,Word2VecModel
from pyspark.mllib.clustering import KMeans, KMeansModel
from math import sqrt

word2vec = Word2Vec()

input = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('Word2Vec Average')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
dict1=pickle.load(open(output+'/output_vector.txt', "rb"))

def transform_data(data):
    data1=re.sub(r'[^\w\s]',' ',data).lower()
    data2=re.sub(' +',' ', data1).strip().split(' ')
    return data2

def mapper(l):
    list_vectors=[]
    if str(l[1]) in dict1.keys():
        return (l[0],dict1[str(l[1])].tolist())
    # return (l[0],list_vectors)

def add_tuples(a, b):
    return tuple(sum(p)/len(p) for p in zip(a,b))

def f(x): return x

def main():
    #Reading the json file
    reviews_data = sqlContext.read.json(input)
    reviews=reviews_data.select('reviewText')
    rdd_data=reviews.rdd.map(lambda line:str(line.reviewText))
    transformed_data=rdd_data.map(transform_data).zipWithIndex()
    transform_list=transformed_data.map(lambda (x,y):(y,x))

    # Transforming the words into vectors
    transformed_data_list=transform_list.flatMapValues(f).map(mapper).filter(lambda l:l is not None)
    reduced_vector=transformed_data_list.reduceByKey(add_tuples)
    avg_vector=reduced_vector.map(lambda (a,b):b).collect()
    dates_rdd=reviews_data.rdd.map(lambda line:str(line.reviewTime)).map(lambda line:line.split(", ")).map(lambda (a,b):b).cache().collect()
    score_rdd=reviews_data.rdd.map(lambda line:str(line.overall)).cache().collect()

    combinedList=zip(avg_vector,score_rdd,dates_rdd)
    combinedRDD=sc.parallelize(combinedList)
    TrainRDD=combinedRDD.filter(lambda (x,y,z):z!='2014').map(lambda (x,y,z):(x,y))
    TestRDD=combinedRDD.filter(lambda (x,y,z):z=='2014').map(lambda (x,y,z):(x,y))

    #Saving testing and training data
    TrainRDD.saveAsPickleFile(output+'/Train_data.average')
    TestRDD.saveAsPickleFile(output+'/Test_data.average')



if __name__ == "__main__":
    main()