import random
import matplotlib.pyplot as plt
import numpy
import scipy.special
import data_arr


class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, hiddennodes2, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        self.lr = learningrate
        self.whi = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.wh2h = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes2))
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.whi, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        hidden_inputs2 = numpy.dot(self.wh2h, hidden_outputs)
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        final_inputs = numpy.dot(self.who, hidden_outputs2)
        final_outputs = self.activation_function(final_inputs)
        output_erros = targets - final_outputs
        hidden_errors2 = numpy.dot(self.who.T, output_erros)
        hidden_errors = numpy.dot(self.wh2h.T, hidden_errors2)

        self.who += self.lr * numpy.dot((output_erros * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs2))
        self.wh2h += self.lr * numpy.dot((hidden_errors2 * hidden_outputs2 * (1 - hidden_outputs2)),
                                         numpy.transpose(hidden_outputs))
        self.whi += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.whi, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        hidden_inputs2 = numpy.dot(self.wh2h, hidden_outputs)
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        final_inputs = numpy.dot(self.who, hidden_outputs2)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def backquery(self, targets_list):
       final_outputs = numpy.array(targets_list, ndmin=2).T

       # calculate the signal into the final output layer
       final_inputs = self.inverse_activation_function(final_outputs)

       # calculate the signal out of the hidden layer
       hidden_outputs2 = numpy.dot(self.who.T, final_inputs)
       # scale them back to 0.01 to .99
       hidden_outputs2 -= numpy.min(hidden_outputs2)
       hidden_outputs2 /= numpy.max(hidden_outputs2)
       hidden_outputs2 *= 0.98
       hidden_outputs2 += 0.01

       # calculate the signal into the hidden layer
       hidden_inputs2 = self.inverse_activation_function(hidden_outputs2)
       hidden_outputs = numpy.dot(self.wh2h.T, hidden_inputs2)
       hidden_outputs -= numpy.min(hidden_outputs)
       hidden_outputs /= numpy.max(hidden_outputs)
       hidden_outputs *= 0.98
       hidden_outputs += 0.01

       hidden_inputs = self.inverse_activation_function(hidden_outputs)
       # calculate the signal out of the input layer
       inputs = numpy.dot(self.whi.T, hidden_inputs)
       # scale them back to 0.01 to .99
       inputs -= numpy.min(inputs)
       inputs /= numpy.max(inputs)
       inputs *= 0.98
       inputs += 0.01

       return inputs



inputnodes = 9
hiddennodes = 25
hiddennodes2 = 40
outputnodes = 10
learningrate = 0.3

inputnodes = 784
hiddennodes = 100
hiddennodes2 = 100
outputnodes = 10

n = neuralNetwork(inputnodes, hiddennodes, hiddennodes2, outputnodes, learningrate)

training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(outputnodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# test
storecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    print(correct_label, "истинный маркер")
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print(label, "ответ сети")
    if label == correct_label:
        storecard.append(1.0)
    else:
        storecard.append(0.0)

storecard_array = numpy.asfarray(storecard)
print(storecard_array)
print("Эффективность сети", storecard_array.sum() / storecard_array.size)

targets = numpy.zeros(outputnodes) + 0.01
label = 3
targets[label] = 0.99
print(targets)
ar = n.backquery(targets)
plt.imshow(ar.reshape(28,28), cmap='Greys', interpolation='None')
plt.show()
#print(ar)


#       n.train(data_arr.lis[nset], data_arr.anw[nset])
# n = neuralNetwork(inputnodes, hiddennodes, hiddennodes2, outputnodes, learningrate)
# for x in range(0, 10000):
#        nset = random.randint(0, len(data_arr.lis) - 1)
#
# print(n.query([0, 0, 1, 0, 0, 1, 0, 0, 0]))
