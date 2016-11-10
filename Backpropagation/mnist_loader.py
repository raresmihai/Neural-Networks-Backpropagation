import cPickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] # transform input image to numpy array of size [784][1] (for easier calculation between layers)
    training_results = [get_vector_from_digit(y) for y in tr_d[1]] # transform each digit into a vector (e.g. 2 -> [0,0,1,0,0,0,0,0,0,0]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def get_vector_from_digit(y):
    vector = np.zeros((10, 1))
    vector[y] = 1.0
    return vector