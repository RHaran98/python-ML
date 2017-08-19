import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score as score
class neuralNetwork:
    def __init__(self,x,y,no_of_classes,no_hidden_layers = 3, learning_rate = 0.01, neurons_per_layer = 5):
        self.data_x = x
        self.data_y = y
        self.no_of_input = len(x[0])
        self.no_of_classes = no_of_classes
        self.x = tf.placeholder("float", shape=[None, self.no_of_input])
        self.y = tf.placeholder("float", shape=[None, self.no_of_classes])
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
        self.biases['layer0'] = tf.Variable(tf.random_normal([ self.neurons_per_layer]))
        for i in range(1,self.no_hidden_layers+1):
            self.weights['layer'+str(i)] = tf.Variable(tf.random_normal([self.neurons_per_layer, self.neurons_per_layer]))
            self.biases['layer' + str(i)] = tf.Variable(tf.random_normal([ self.neurons_per_layer]))
        self.weights['output'] = tf.Variable(tf.random_normal([ self.neurons_per_layer,self.no_of_classes]))
        self.biases['output'] = tf.Variable(tf.random_normal([ self.no_of_classes]))

    def define_graph(self):
        self.output = self.x         #Input
        with tf.device('/gpu:0'):
            for i in range(self.no_hidden_layers+1):
                self.output = tf.add(tf.matmul(self.output,self.weights['layer'+str(i)]),self.biases['layer'+str(i)])
                self.output = tf.nn.relu(self.output)
            self.output = tf.matmul(self.output, self.weights['output'])+self.biases['output']

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def train(self,epochs):
        self.init_weights()
        self.define_graph()
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epochs):
                #for point,label in zip(self.data_x,self.data_y):
                _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: self.data_x, self.y: self.data_x})
                if i % 100 == 0:
                    print "Epoch : {0} Cost : {1}\n".format(i+1,c)
    def pred(self,x):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pred = sess.run(self.output,feed_dict={ self.x:x})
            pred_args = sess.run(tf.arg_max(pred,dimension=1))
        return pred,pred_args
