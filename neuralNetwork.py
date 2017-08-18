import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets
from matplotlib import pyplot as plt

class neuralNetwork:
    def __init__(self,x,y,no_of_classes,no_hidden_layers = 3, learning_rate = 0.01, neurons_per_layer = 10):
        self.data_x = x
        self.data_y = y
        self.no_of_input = len(x[0])
        self.no_of_classes = no_of_classes
        self.x = tf.placeholder("float", [None, self.no_of_input])
        self.y = tf.placeholder("float", [None, self.no_of_classes])
        self.no_hidden_layers = no_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.weights = {}
        self.biases = {}
        self.output = None      #
        self.cost = None        # Will be initialized in method define_graph
        self.optimizer = None   #

    def init_weights(self):
        self.weights['layer0'] = tf.Variable(tf.random_normal([self.no_of_input, self.neurons_per_layer]))
        self.biases['layer0'] = tf.Variable(tf.random_normal([self.no_of_input, self.neurons_per_layer]))
        for i in range(1,self.no_hidden_layers+1):
            self.weights['layer'+str(i)] = tf.Variable(tf.random_normal([self.neurons_per_layer, self.neurons_per_layer]))
            self.biases['layer' + str(i)] = tf.Variable(tf.random_normal([self.neurons_per_layer, self.neurons_per_layer]))
        self.weights['output'] = tf.Variable(tf.random_normal([self.neurons_per_layer, self.neurons_per_layer]))
        self.weights['output'] = tf.Variable(tf.random_normal([self.neurons_per_layer, self.no_of_classes]))

    def define_graph(self):
        self.output = self.x         #Input
        for i in range(self.no_hidden_layers+1):
            print "Shape : ",self.output.get_shape()
            print "Before op : ",self.weights['layer' + str(i)].get_shape()
            self.output = tf.add(tf.matmul(self.output,self.weights['layer'+str(i)]),self.biases['layer'+str(i)])
            print "After op : ",self.weights['layer' + str(i)].get_shape(),'\n'
            self.output = tf.tanh(self.output)
        self.output = tf.matmul(self.output,weights['output'])+biases['output']
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    def train(self,epochs):
        self.init_weights()
        self.define_graph()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epochs):
                for point,label in zip(self.data_x,self.data_y):
                    _, c = sess.run([optimizer, cost], feed_dict={self.x: point , self.y: label})
                if i%100==0:
                    print "Cost : " + str(c)




np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()

nn = neuralNetwork(X,y,2)
nn.train(500)