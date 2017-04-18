__author__ = 'BANDEEP SINGH'
from pyspark import SparkConf, SparkContext, SQLContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
import sys, operator, json, string,re,pickle,ast
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.mllib.feature import Word2Vec,Word2VecModel
from pyspark.mllib.clustering import KMeans, KMeansModel
from math import sqrt
import scipy.sparse as sps
from pyspark.mllib.linalg import Vectors

word2vec = Word2Vec()

input = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('Question 7')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
dict1=pickle.load(open(output+'/cluster_data.txt', "rb"))

def transform_data(data):
    data1=re.sub(r'[^\w\s]',' ',data).lower()
    data2=re.sub(' +',' ', data1).strip().split(' ')
    return data2

def mapper(l):
    list_vectors=[]
    list=l[1]
    for element in list:
        if str(element) in dict1.keys():
            list_vectors.append(dict1[str(element)])
    return (l[0],list_vectors)

def add_tuples(a, b):
    return tuple(sum(p)/len(p) for p in zip(a,b))

def f(x): return x

def transform_vector(l):
    list2=l[1]
    length=len(list2)
    dict1={}
    for element in list2:
        dict1[element]=float(list2.count(element))/length
    sv1 = Vectors.sparse(2000, dict1)
    return sv1

def main():
    #Reading the json file
    reviews_data = sqlContext.read.json(input)
    reviews=reviews_data.select('reviewText')
    rdd_data=reviews.rdd.map(lambda line:str(line.reviewText))
    transformed_data=rdd_data.map(transform_data).zipWithIndex()
    transform_list=transformed_data.map(lambda (x,y):(y,x))

    #Transforming thw words into cluster undex
    reduced_vector=transform_list.map(mapper).filter(lambda l:l is not None).map(transform_vector)

    normalized_vector=reduced_vector.collect()
    dates_rdd=reviews_data.rdd.map(lambda line:str(line.reviewTime)).map(lambda line:line.split(", ")).map(lambda (a,b):b).cache().collect()
    score_rdd=reviews_data.rdd.map(lambda line:str(line.overall)).cache().collect()

    combinedList=zip(normalized_vector,score_rdd,dates_rdd)
    combinedRDD=sc.parallelize(combinedList).cache()
    TrainRDD=combinedRDD.filter(lambda (x,y,z):z!='2014').map(lambda (x,y,z):(x,y))
    TestRDD=combinedRDD.filter(lambda (x,y,z):z=='2014').map(lambda (x,y,z):(x,y))

    #Saving Test and Train data
    TrainRDD.saveAsPickleFile(output+'/Train_data.BOF')
    TestRDD.saveAsPickleFile(output+'/Test_data.BOF')

if __name__ == "__main__":
    main()