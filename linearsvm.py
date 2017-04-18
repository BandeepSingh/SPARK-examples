from pyspark import SparkConf, SparkContext,SQLContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


def main():
    data=np.load('C:\\spark-1.5.1-bin-hadoop2.6\\bandeep\\Assignment6\\forest_data.npz')
    # print data['label_training']
    # Unnormalized
    training_data=data['data_training']
    y_training=data['label_training']
    y_validation=data['label_val']
    validation_data=data['data_val']
    acc=[]
    c=[pow(10,-12),pow(10,-11),pow(10,-10),pow(10,-9),pow(10,-8),pow(10,-7),pow(10,-6),pow(10,-5),pow(10,-4),pow(10,-3),pow(10,-2),pow(10,-1),pow(10,-0),pow(10,1),pow(10,2),pow(10,3),pow(10,4),pow(10,5)]
    for x in c:
        clf = svm.LinearSVC(C = x)
        clf.fit(training_data,y_training)
        predicted=clf.predict(validation_data)
        acc.append(accuracy_score(y_validation, predicted))
        print acc
    plt.plot(c,acc,label='UnNormalized')
    plt.axis([c[0],c[-1], 0, 1])
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.legend()

    # Normalized SVM
    acc_norm=[]
    y_training=data['label_training']
    y_validation=data['label_val']
    training_data=preprocessing.scale(data['data_training'])
    validation_data=preprocessing.scale(data['data_val'])
    c=[pow(10,-12),pow(10,-11),pow(10,-10),pow(10,-9),pow(10,-8),pow(10,-7),pow(10,-6),pow(10,-5),pow(10,-4),pow(10,-3),pow(10,-2),pow(10,-1),pow(10,-0),pow(10,1),pow(10,2),pow(10,3),pow(10,4),pow(10,5)]
    for x in c:
        clf = svm.LinearSVC(C = x)
        clf.fit(training_data,y_training)
        predicted=clf.predict(validation_data)
        acc_norm.append(accuracy_score(y_validation, predicted))
        print acc_norm
    plt.plot(c,acc_norm,label='Normalized')
    plt.axis([c[0],c[-1], 0, 1])
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
  main()
