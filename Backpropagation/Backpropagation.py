import random
import numpy as np

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

def learn():
    learning_rate = 0.5
    lmbda = 5.0
    number_of_iterations = 10
    mini_batch_size = 10
    n = 50000
    biases = [np.random.randn(y, 1) for y in [100,10]] #[[100][1],[10][1]]
    weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in [(784,100),(100,10)]] #[[100][784],[10][100]]

    for iteration in xrange(number_of_iterations):
        random.shuffle(training_data)
        mini_batches = [
            training_data[k:k+mini_batch_size]
            for k in xrange(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            nabla_b = [np.zeros(b.shape) for b in biases] #[[100][1] , [10][1]] 
            nabla_w = [np.zeros(w.shape) for w in weights] # [[100][784],[10][100]] mini batch total weight gradient : tells us how to modify the weights that come into each neuron
            for x, y in mini_batch:
                bias_gradient, weight_gradient = backpropagation(x, y,biases,weights) # gradients for just one input
                nabla_b = [nb+bg for nb, bg in zip(nabla_b, bias_gradient)] #gradients for the total mini batch (sum of 10 gradients)
                nabla_w = [nw+wg for nw, wg in zip(nabla_w, weight_gradient)]

            #weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(weights, nabla_w)]
            weights = [(1-learning_rate*(lmbda/n))*w-(learning_rate/len(mini_batch))*nw    #regularization - course 5 , slide 30
                            for w, nw in zip(weights, nabla_w)]
            biases = [b-(learning_rate/len(mini_batch))*nb
                           for b, nb in zip(biases, nabla_b)]

        number_of_correct_outputs = test_accuracy(test_data,biases,weights)
        accuracy = float(number_of_correct_outputs)/100
        print "Iteration {}: {} %".format(iteration, accuracy)
    
def backpropagation(x,y,biases,weights):       
        z1 = x		
        activation_1 = z1

        z2 = np.dot(weights[0],activation_1) + biases[0] # [100][784] * [784][1] -> [100][1] : input of each neuron from layer 2
        activation_2 = sigmoid(z2) # [100][1] activation of each neuron from layer 2

        z3 = np.dot(weights[1],activation_2) + biases[1] # [10][100] * [100][1] -> [10][1] : input of each neuron from layer 3
        activation_3 = softmax(z3) # [10][1] activation of each neuron(digit) from layer 3 

        delta_23 = activation_3-y # [10][1] error on (last) layer 3 (cross entropy) 
        bias_gradient23 = delta_23 
        weight_gradient23 = np.dot(delta_23,activation_2.transpose()) # [10][1] * [1][100] -> [10][100] gradient for weights from layer 2 to layer 3

        delta_12 = np.dot(weights[1].transpose(), delta_23) * sigmoid_prime(z2) # [100][10] * [10][1] ->  [100][1] error on layer 2 (backpropagated using the weights from layer 2 to layer 3) - course 3, page 27
        bias_gradient12 = delta_12
        weight_gradient12 = np.dot(delta_12,activation_1.transpose()) # [100][1] * [1][784] -> [100][784] gradient for weights from layer 1 to layer 2

        bias_gradient = [bias_gradient12,bias_gradient23]
        weight_gradient = [weight_gradient12,weight_gradient23]

        return (bias_gradient,weight_gradient)

def test_accuracy(test_data,biases,weights):
    number_of_correct_outputs = 0
    for (x,y) in test_data:
        output = find_output(x,biases,weights) # returns a vector [10] where each element(digit) has a probability of being the output
        digit = np.argmax(output) # get the digit with the highest probability
        if digit == y:
            number_of_correct_outputs += 1
    return number_of_correct_outputs

def find_output(x,biases,weights):
    activation_2 = sigmoid(np.dot(weights[0],x)+biases[0])
    activation_3 = softmax(np.dot(weights[1],activation_2)+biases[1])
    return activation_3

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    exp_sum = sum(np.exp(zk) for zk in z)
    return np.exp(z)/exp_sum

learn()