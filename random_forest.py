from pyspark import SparkConf, SparkContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
import sys, pickle,math
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('Random Forest')
sc = SparkContext(conf=conf)

input = sys.argv[1]
# output = sys.argv[2]

def parseInput(line):
    return LabeledPoint(float(line[1]),line[0])

def main():
    #Reading the test and train  files
    trainData  = sc.pickleFile(input+'/Train_data.average/part-00000')
    testData = sc.pickleFile(input+'/Test_data.average/part-00000')
    parsedData=trainData.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)
    parsedTestData = testData.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)
    numTrees=[3,5,10]
    bestmaxBins=[5,10,15]
    BestError=1000000

    #Cross validation
    for x in bestmaxBins:
        for y in numTrees:
            (Train_RDD,Valid_RDD)=trainData.randomSplit([1,2],10L)
            parsed_input=Train_RDD.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)
            parsed_valid = Valid_RDD.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)
            model = RandomForest.trainRegressor(parsed_input, categoricalFeaturesInfo={},numTrees=y, featureSubsetStrategy="auto",impurity='variance', maxDepth=4, maxBins=x)
            predictions = model.predict(parsed_valid.map(lambda x: x.features))
            labelsAndPredictions = parsed_valid.map(lambda lp: lp.label).zip(predictions)
            validationErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(parsed_valid.count())
            RMSE=math.sqrt(validationErr)
            if RMSE<BestError:
                    BestError=RMSE
                    bestmaxBins=x
                    bestnumTrees=y
    #Finding Test error
    model = RandomForest.trainRegressor(parsedData, categoricalFeaturesInfo={},numTrees=bestnumTrees, featureSubsetStrategy="auto",impurity='variance', maxDepth=4, maxBins=bestmaxBins)
    predictions = model.predict(parsedTestData.map(lambda x: x.features))
    labelsAndPredictions = parsedTestData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(parsedTestData.count())
    RMSE_test=math.sqrt(testErr)

    #Reporting validation and test error
    print("Best Root Mean Squared Error Validation= " + str(BestError))
    print("Best Root Mean Squared Error Test= " + str(RMSE_test))

if __name__ == "__main__":
    main()
