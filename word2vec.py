from pyspark import SparkConf, SparkContext, SQLContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
import sys, operator, json, string,re, pickle,ast
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.mllib.feature import Word2Vec,Word2VecModel
from pyspark.mllib.clustering import KMeans, KMeansModel
from math import sqrt

word2vec = Word2Vec()

input = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('Word2Vec')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def transform_data(data):
    data1=re.sub(r'[^\w\s]',' ',data).lower()
    data2=re.sub(' +',' ', data1).strip().split(' ')
    return data2


def error(clusters,point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))


def main():
    #Reading the json file
    reviews_data = sqlContext.read.json(input)
    reviews=reviews_data.select('reviewText')
    rdd_data=reviews.rdd.map(lambda line:str(line.reviewText))
    transformed_data=rdd_data.map(transform_data).cache()
    #Transforming the words
    model = word2vec.fit(transformed_data)
    #Finding distinct words
    unique_words=transformed_data.flatMap(lambda l:l).map(lambda l:str(l)).distinct()
    # print unique_words.collect()
    dict1={}
    for a in unique_words.collect():
        try:
            dict1[a]=model.transform(a)
        except Exception:
            pass

    # Saving word2vec model
    pickle.dump(dict1, open(output+'\output_vector_sample.txt', "wb"))

    # dict2=pickle.load(open(output+'/output4.txt', "rb"))
    #finding synonyms
    # synonyms = model.findSynonyms('happy', 10)
    # print synonyms
    feature_vectors=dict1.values()
    feature_vectors_rdd=sc.parallelize(feature_vectors)
    clusters = KMeans.train(feature_vectors_rdd, 2000, maxIterations=1,runs=1, initializationMode="random")
    # WSSSE=feature_vectors_rdd.map(lambda point: error(clusters,point)).reduce(lambda x, y: x + y)
    # print("Within Set Sum of Squared Error = " + str(WSSSE))
    cluster_predictions={}
    for key in dict1.keys():
        cluster_predictions[key]=clusters.predict(dict1[key])

    # Saving word to cluster index model
    pickle.dump(cluster_predictions,open(output+'/cluster_data.txt', "wb"))

if __name__ == "__main__":
    main()