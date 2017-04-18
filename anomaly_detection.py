# anomaly_detection.py
from pyspark import SparkConf, SparkContext, SQLContext
# SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType,FloatType,ArrayType
import sys
#import operator

input = sys.argv[1]
conf = SparkConf().setAppName('anomaly_detection')
sc = SparkContext(conf=conf)
conf.set("spark.storage.memoryFraction", "0.40")
sqlCt = SQLContext(sc)

class AnomalyDetection():

    def readData(self, filename):
        self.rawDF = sqlCt.read.parquet(filename).cache()

    def cat2Num(self, df, indices):
        distinct_values1=df.select(df.rawFeatures[indices[0]]).distinct().collect()
        len_distinct=df.select(df.rawFeatures[indices[0]]).distinct().count()
        distinct_values2=df.select(df.rawFeatures[indices[1]]).distinct().collect()
        len_distinct2=df.select(df.rawFeatures[indices[1]]).distinct().count()
        hot_encoding1={}
        for i ,value in enumerate(distinct_values1):
            hot_encoding1[value[0]]=[0]*len_distinct
            hot_encoding1[value[0]][i]=1

        hot_encodingBC=sc.broadcast(hot_encoding1)
        hot_encoding2={}
        for i ,value in enumerate(distinct_values2):
            hot_encoding2[value[0]]=[0]*len_distinct2
            hot_encoding2[value[0]][i]=1

        hot_encoding2BC=sc.broadcast(hot_encoding2)

        def transform(raw_feat):
            features=[]
            features.extend(hot_encodingBC.value[raw_feat[0]])
            features.extend(hot_encoding2BC.value[raw_feat[1]])
            features.extend([raw_feat[2]])
            new_list=[float(x) for x in features]
            # map1=list(map(float,features))
            return new_list

        slen=udf(transform, ArrayType(FloatType()))
        df1=df.withColumn("features", slen(df.rawFeatures))
        return df1

    def addScore(self, df):
        df_features=df.select(df.prediction)
        df_count=df_features.collect()
        dict1={}
        for x in df_count:
            if x[0] in dict1.keys():
                dict1[x[0]]=dict1[x[0]]+1
            else:
                dict1[x[0]]=1
        dictBC=sc.broadcast(dict1)
        def cal_score(data):
            N_max=float(sorted(dictBC.value.values(),reverse=True)[0])
            N_min=float(sorted(dictBC.value.values())[0])
            N_x=float(dict1[data[0]])
            try:
                score_x = float(N_max - N_x)/(N_max - N_min)
            except Exception:
                score_x=0.0
            return score_x
        slen=udf(cal_score, FloatType())
        df1=df.withColumn("score", slen(df.prediction))
        return df1

    def detect(self, k, t):
        #Encoding categorical features using one-hot.
        df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
        df1.show()

        #Clustering points using KMeans
        features = df1.select("features").rdd.map(lambda row: row[0]).cache()
        # print features.collect()
        model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

        # #Adding the prediction column to df1
        modelBC = sc.broadcast(model)
        predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
        df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
        df2.show()

        #Adding the score column to df2; The higher the score, the more likely it is an anomaly
        df3 = self.addScore(df2).cache()
        df3.show()

        return df3.where(df3.score > t)

    def readToyData(self):
        data = [(0, ["http", "udt", 0.4]), \
                (1, ["http", "udf", 0.5]), \
                (2, ["http", "tcp", 0.5]), \
                (3, ["ftp", "icmp", 0.1]), \
                (4, ["http", "tcp", 0.4])]
        schema = ["id", "rawFeatures"]
        self.rawDF = sqlCt.createDataFrame(data, schema)

if __name__ == "__main__":
    ad = AnomalyDetection()
    ad.readData(input)
    anomalies = ad.detect(8, 0.97)
    print anomalies.count()
    anomalies.show()