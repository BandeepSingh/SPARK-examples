from pyspark import SparkConf, SparkContext,SQLContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def main():
    data=np.load('C:\\spark-1.5.1-bin-hadoop2.6\\bandeep\\Assignment6\\forest_data.npz')
    forest_size=[10,20,50,100]
    sampling_rate=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # sampling_rate=[0.1]
    y_training=data['label_training']
    acc={}
    for x in sampling_rate:
        acc_temp=[]
        for y in forest_size:
            clf = BaggingClassifier(base_estimator=None,max_samples=x,n_estimators=y)
            clf.fit(data['data_training'],y_training)
            predicted=clf.predict(data['data_val'])
            acc_temp.append(accuracy_score(data['label_val'], predicted))
            print acc_temp
        acc[x]=acc_temp
    i=0
    for x in acc.keys():
        plt.semilogx(forest_size,acc[x],label='Samplin rate:'+str(sampling_rate[i]))
        i+=1
    plt.xlabel('Forest Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
  main()

