import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
import math
from collections import Counter

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, :])
    # print(data)
    return data[:, :-1], data[:, -1]

X,y=create_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class NaiveBayes:
    '''
    高斯贝叶斯：对于连续变量，无法通过统计频率估计概率，因此，如果该变量可以认为服从正态分布，则可以用高斯分布来估计概率。
    如果该变量服从其他连续分布，也可以用其他连续分布计算概率。



    利用训练集估计高斯分布的均值和方差参数。将测试集中的样本带入高斯分布函数中计算概率。

    '''
    def __init__(self) -> None:
        self.model=None

    @staticmethod
    def mean(X):
        return sum(X)/float(len(X))
    
# 计算方差
    def stdev(self,X):
        avg=self.mean(X)
        return math.sqrt(sum(pow(x-avg,2) for x in X)/float(len(X)))
    


    def gaussian_probability(self,X,mean,stdev):
        exponent=math.exp(-(math.pow(X-mean,2)/(2*math.pow(stdev,2))))
        return (1/(math.sqrt(2*math.pi)*stdev))*exponent
    
    def summarize(self,train_data):
        summaries=[(self.mean(i),self.stdev(i)) for i in  zip(*train_data)]
        return summaries

    def fit(self,X,y):
        '''
        分别求出数学期望和标准差
        '''    
    
        labels=list(set(y))

        data={label:[] for label in labels}

        for f,label in  zip(X,y):
            data[label].append(f)
        
        self.model={
            label:self.summarize(value)
            for label,value in data.items()
        }

        return "gussain NB train done"
    
    #计算概率

    def calculate_probability(self,input_data):
        probabilities={}

        for label,value in self.model.items():
            probabilities[label]=1
            
            for i in range(len(value)):
                mean,stdev=value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
        return probabilities
    
    #类别
    def predict(self,X_test):
        label=sorted(
            self.calculate_probability(X_test).items(),
            key=lambda d:d[-1])[-1][0]
        return label
    
    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))


model=NaiveBayes()
model.fit(X_train,y_train)

print(model.score(X_test,y_test))











            