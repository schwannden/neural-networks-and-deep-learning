import mnist_loader
import network
import time
start_time = time.clock()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print "loading data: ", time.clock() - start_time
net = network.Network([784, 30, 10])
start_time = time.clock()
net.SGD(training_data, 50, 10, 3.0, test_data=test_data)
print "training data: ", time.clock() - start_time
