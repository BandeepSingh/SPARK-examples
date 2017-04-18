from pyspark import SparkConf, SparkContext, Row
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
import sys,math,re

movie_dir = sys.argv[1]

conf = SparkConf().setAppName('Movie Recommendations ALS')
sc = SparkContext(conf=conf)
# sqlContext = SQLContext(sc)

def convert_val(a,b):
    temp=[]
    for p,q in b:
        for k,v in b:
            diff=q-v
            temp.append(((p,k),(diff,1)))
    return temp

def add_tuples(x, y):
    return tuple(sum(p) for p in zip(x, y))


def main():
    ratings = sc.textFile(movie_dir+'/MovieLens100K_train.txt')
    movies=sc.textFile(movie_dir+'/MovieLens100K_test.txt')
    ratingsRDD=ratings.map(lambda line: re.split(r'\t+', line)).map(lambda (UserID,MovieID,Rating,uniqueID):(int(UserID),(int(MovieID),float(Rating)))).cache()
    RDD1=ratingsRDD.groupByKey().mapValues(list)
    RDD2=RDD1.map(lambda l:convert_val(l[0],l[1])).flatMap(lambda l:l)
    RDD3=RDD2.reduceByKey(add_tuples).map(lambda (a,b):(a,(b[0]/b[1],b[1])))
    testRDD=movies.map(lambda line: re.split(r'\t+', line)).map(lambda (UserID,MovieID,Rating,uniqueID):((int(UserID),int(MovieID)),float(Rating))).cache()
    testdataRDD=testRDD.map(lambda (a,b):a)
    joined1=testdataRDD.join(ratingsRDD).map(lambda (a,b):((b[0],b[1][0]),(a,b[1][1])))
    joined2=joined1.join(RDD3)
    joined3=joined2.map(lambda (a,b):((b[0][0],a[0]),((b[1][0]+b[0][1])*b[1][1],b[1][1])))
    final_map=joined3.reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1])).map(lambda (a,b):(a,float(b[0])/b[1]))
    final_join=testRDD.join(final_map)
    MSE=final_join.map(lambda (usermovie,ratings):(ratings[1]-ratings[0])**2).mean()
    RMSE=math.sqrt(MSE)
    print("Root Mean Squared Error = " + str(RMSE))

if __name__ == "__main__":
    main()


