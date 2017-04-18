from pyspark import SparkConf, SparkContext,SQLContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def main():
    data=np.load('C:\\spark-1.5.1-bin-hadoop2.6\\bandeep\\Assignment6\\forest_data.npz')
    c=[pow(10,-12),pow(10,-11),pow(10,-10),pow(10,-9),pow(10,-8),pow(10,-7),pow(10,-6),pow(10,-5),pow(10,-4),pow(10,-3),pow(10,-2),pow(10,-1),pow(10,0),pow(10,1),pow(10,2),pow(10,3),pow(10,4),pow(10,5)]
    G=[pow(10,-8),pow(10,-7),pow(10,-6),pow(10,-5),pow(10,-4),pow(10,-3),pow(10,-2),pow(10,-1)]
    acc={}
    training_data=data['data_training']
    y_training=data['label_training']
    y_validation=data['label_val']
    validation_data=data['data_val']
    for y in G:
        acc_temp=[]
        for x in c:
            clf = svm.SVC(kernel='rbf',C = x,gamma=y)
            clf.fit(training_data[0:2000],y_training[0:2000])
            predicted=clf.predict(validation_data)
            acc_temp.append(accuracy_score(y_validation, predicted))
            print acc_temp
        acc[y]=acc_temp
    i=0
    for x in acc.keys():
        plt.semilogx(c,acc[x],label="Gamma: "+str(G[i]))
        plt.axis([c[0],c[-1], 0, 1])
        i+=1
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
  main()

