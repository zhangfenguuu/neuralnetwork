import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network
net = network.Network([784, 30, 10])#　　网络结构
evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = net.SGD(training_data, 30, 10, 0.5,lmbda = 5.0,evaluation_data=validation_data,monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,monitor_training_accuracy=True,monitor_training_cost=True)
