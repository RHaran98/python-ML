import numpy as np
import random
from matplotlib import pyplot as plt

class linearRegression_np:
    def __init__(self,X_train,Y_Train,alpha = 0.1):
        self.X_Train = self.add_bias(X_train)
        self.Y_Train = Y_Train
        self.weights = [random.random()]*(len(self.X_Train[0]))
        self.alpha = alpha

    def add_bias(self,list):
        new_list = []
        for i in list:
            new_list.append( [1] + i)
        return new_list

    def pred(self,X):
        return sum(np.multiply(self.weights,X))

    def derivative_cost(self,j):
        m = len(self.X_Train)
        cost = 0
        for i in range(m):
            cost += (self.pred(self.X_Train[i])-self.Y_Train[i])*self.X_Train[i][j]
        cost/=m
        return cost

    def train(self):
        m = len(self.X_Train[0])
        for i in range(m):
            self.weights[i] -= self.alpha * self.derivative_cost(i)
        #self.bias-=self.bias_cost()

    def plot(self):
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = self.weights[0] + self.weights[1] * x_vals
        plt.plot(x_vals,y_vals,'--')
        plt.plot(self.X_Train,self.Y_Train,'ro')
        plt.show()
