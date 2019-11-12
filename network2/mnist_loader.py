import _pickle as cPickle
import gzip
import numpy as np
import sys
import imp
imp.reload(sys)
def load_data():
    f=gzip.open('/home/scott/Desktop/data/neural-networks-and-deep-learning/data/mnist.pkl.gz','rb')
    training_data,validation_data,test_data=cPickle.load(f,encoding='iso-8859-1')
    f.close()
    return (training_data,validation_data,test_data)

def load_data_wrapper():
    tr_d,va_d,te_d=load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)  #zip对象　每个元素为含两个元素的元组　第一个元素是input 784,1 的nparray 第二个元素是result 10,1　的nparray
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])                   #zip对象　每个元素为含两个元素的元组　第一个元素是input 784,1 的nparray 第二个元素是０－９整数
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
