from pyspark import SparkConf, SparkContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
import sys, operator,pickle,math
from pyspark.mllib.regression import LabeledPoint,LinearRegressionWithSGD

conf = SparkConf().setAppName('Linear regression')
sc = SparkContext(conf=conf)

input = sys.argv[1]
# output = sys.argv[2]

def parseInput(line):
    return LabeledPoint(float(line[1]),line[0])

def main():
    #reading test and train data
    trainData  = sc.pickleFile(input+'/Train_data_unnormalized.pickle/part-00000')
    testData = sc.pickleFile(input+'/Test_data_unnormalized.pickle/part-00000')
    parsedData=trainData.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)
    parsedTestData = testData.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)

    numIterations = 100
    stepSize=[0.1,10,20]
    BestError=1000000

    BestStep=0
    BestSplit=[]
    splits=[[1,2],[1,3]]

    #Cross Validation
    for x in stepSize:
        for y in splits:
            (Train_RDD,Valid_RDD)=trainData.randomSplit(y,10L)
            parsed_input=Train_RDD.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)
            parsed_valid=Valid_RDD.map(parseInput).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)
            try:
                model = LinearRegressionWithSGD.train(parsed_input,iterations=numIterations,step=x)
                valuesAndPreds = parsed_valid.map(lambda p: (p.label, model.predict(p.features)))
                MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
                RMSE=math.sqrt(MSE)
            except Exception:
                pass
            if RMSE<BestError:
                BestError=RMSE
                BestStep=x
                BestSplit=y


    #Finding test error

    model = LinearRegressionWithSGD.train(parsedData,iterations=numIterations,step=BestStep)
    valuesAndPreds = parsedTestData.map(lambda p: (p.label, model.predict(p.features)))
    MSE_test = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    RMSE_test=math.sqrt(MSE_test)


    print("Best Root Mean Squared Error Validation = " + str(BestError))
    print("Best Root Mean Squared Error Test= " + str(RMSE_test))
    print("Best StepSize = " + str(BestStep))
    print(BestSplit)

if __name__ == "__main__":
    main()