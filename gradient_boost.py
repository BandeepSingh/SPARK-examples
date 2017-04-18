from pyspark import SparkConf, SparkContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
import sys, pickle,math
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Random Forest')
sc = SparkContext(conf=conf)

input = sys.argv[1]
# output = sys.argv[2]

def parseInput(line):
    return LabeledPoint(float(line[1]),line[0])

def main():
    #Reading train and test data
    trainData  = sc.pickleFile(input+'/Train_data.average/part-00000')
    testData = sc.pickleFile(input+'/Test_data.average/part-00000')
    parsedData=trainData.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)
    parsedTestData = testData.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0).cache()
    model = GradientBoostedTrees.trainRegressor(parsedData,categoricalFeaturesInfo={}, numIterations=1)
    predictions = model.predict(parsedTestData.map(lambda x: x.features))
    labelsAndPredictions = parsedTestData.map(lambda lp: lp.label).zip(predictions)
    validationErr = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(parsedTestData.count())
    parsedTestData.unpersist()
    RMSE=math.sqrt(validationErr)

    print("Root Mean Squared Error Test= " + str(RMSE))

if __name__ == "__main__":
    main()
