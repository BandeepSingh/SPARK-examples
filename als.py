from pyspark import SparkConf, SparkContext, SQLContext,Row
# SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
from pyspark.mllib.recommendation import ALS
import sys,math,re
import matplotlib.pyplot as plt

movie_dir = sys.argv[1]

conf = SparkConf().setAppName('Movie Recommendations ALS')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

def main():
    ratings = sc.textFile(movie_dir+'/MovieLens100K_train.txt')
    movies=sc.textFile(movie_dir+'/MovieLens100K_test.txt')
    ratingsRDD=ratings.map(lambda line: re.split(r'\t+', line)).map(lambda (UserID,MovieID,Rating,uniqueID):(int(UserID),int(MovieID),float(Rating)))
    # print ratingsRDD.take(10)
    testdataRDD=movies.map(lambda line: re.split(r'\t+', line)).map(lambda (UserID,MovieID,Rating,uniqueID):((int(UserID),int(MovieID)),float(Rating))).cache()
    moviesRDD=testdataRDD.map(lambda (a,b):a)
    # print moviesRDD.take(10)

    training=ratingsRDD
    rank = [2, 4, 8, 16, 32, 64, 128, 256]
    lmbda = [0.01,0.1]
    numIter = 10
    # BestError=1
    maps={}
    for x in lmbda:
        MSE=[]
        for y in rank:
            model = ALS.train(training, y, numIter, x)
            predictions = model.predictAll(moviesRDD)
            predictions_filter=predictions.map(lambda r: ((r[0], r[1]), r[2]))
            ratesAndPreds = testdataRDD.join(predictions_filter)
            MSE_tmp = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
            RMSE=math.sqrt(MSE_tmp)
            MSE.append(RMSE)
            # if RMSE<BestError:
            #     BestError=RMSE
        maps[x]=MSE
    # print("Root Mean Squared Error = " + str(BestError))
    print maps
    for x in maps.keys():
        plt.plot(rank,maps[x],label="Regularizer "+str(x))
        plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

