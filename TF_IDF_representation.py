from pyspark import SparkConf, SparkContext, SQLContext
# SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
import sys, operator, json, string, nltk,re
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from nltk.corpus import stopwords
from pyspark.mllib.feature import HashingTF,IDF
from pyspark.mllib.feature import Normalizer
import pickle
# path to the nltk data directory.
nltk.data.path.append("/cs/vml2/avahdat/CMPT733_Data_Sets/Assignment3/nltk_data")
stop_words = set(stopwords.words("english"))

input = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('TF-IDF Representation')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def transform_data(data):
    data1=re.sub(r'[^\w\s]',' ',data).lower()
    data2=re.sub(' +',' ', data1).strip().split(' ')
    final_data=[word for word in data2 if word not in stop_words]
    return final_data


def main():
    #Reading the json file
    reviews_data = sqlContext.read.json(input)
    reviews=reviews_data.select('reviewText')
    reviews_rdd=reviews.rdd.cache()
    rdd_data=reviews_rdd.map(lambda line:str(line.reviewText))
    transformed_data=rdd_data.map(transform_data)

    #Finding Tf-IDF representation
    hashingTF = HashingTF()
    tf = hashingTF.transform(transformed_data)
    tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf).collect()
    # Normalization
    # tfidf = idf.transform(tf)
    # normalizer1 = Normalizer()
    # normalized_vector=normalizer1.transform(tfidf).collect()

    score_rdd=reviews_data.rdd.map(lambda line:str(line.overall)).cache().collect()
    dates_rdd=reviews_data.rdd.map(lambda line:str(line.reviewTime)).map(lambda line:line.split(", ")).map(lambda (a,b):b).cache().collect()
    combinedList=zip(tfidf,score_rdd,dates_rdd)
    combinedRDD=sc.parallelize(combinedList).cache()
    TrainRDD=combinedRDD.filter(lambda (x,y,z):z!='2014').map(lambda (x,y,z):(x,y))
    TestRDD=combinedRDD.filter(lambda (x,y,z):z=='2014').map(lambda (x,y,z):(x,y))

    #Saving test and training data
    TrainRDD.saveAsPickleFile(output+'/Train_data_unnormalized.pickle')
    TestRDD.saveAsPickleFile(output+'/Test_data_unnormalized.pickle')
    # TrainRDD.saveAsPickleFile(output+'/Train_data_normalized.pickle')
    # TestRDD.saveAsPickleFile(output+'/Test_data_normalized.pickle')


if __name__ == "__main__":
    main()