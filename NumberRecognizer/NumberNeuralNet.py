
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class NumberNeuralNet():

    def __init__(self, learning_rate, iterations, hidden_layers=[20,20]):
        if (len(hidden_layers) != 2):
            hidden_layers = [20, 20]
        
        self.prepare_data(1, hidden_layers)

        for i in range(8):
            self.train_network(self.X[i], self.Y[i], learning_rate, iterations)



    #read data and prepare for learning
    def prepare_data(self, test_cases, hidden_layers):
        data = pd.read_csv('train.csv')
        data = np.array(data)
        np.random.shuffle(data)

        #setting up dimensions of network layers
        m, n = data.shape
        self.layers = [n-1, hidden_layers[0], hidden_layers[1], 10]

        #do this instead of handling errors
        if (test_cases < 1):
            test_cases = 1
        elif (test_cases >= m):
            test_cases = m - 1

        self.examples = m - test_cases

        #get training datasets
        self.X = []
        self.Y = []

        for i in range(7, -1, -1):
            d = data[test_cases + 5000 * (7 - i): m - (5000 * i)].T
            self.X.append(d[1:n] / 255.)
            self.Y.append(d[0])

        #get test dataset
        d_train = data[0:test_cases].T
        self.X_test = d_train[1:n]
        self.X_test = self.X_test / 255.
        self.Y_test = d_train[0]

        self.init_params()

        

    #setting weights and biases with random values
    def init_params(self):
        W1 = np.random.rand(self.layers[1], self.layers[0]) - 0.5
        b1 = np.random.rand(self.layers[1], 1) - 0.5
        W2 = np.random.rand(self.layers[2], self.layers[1]) - 0.5
        b2 = np.random.rand(self.layers[2], 1) - 0.5
        W3 = np.random.rand(self.layers[3], self.layers[2]) - 0.5
        b3 = np.random.rand(self.layers[3], 1) - 0.5
        self.W = [W1, W2, W3]
        self.b = [b1, b2, b3]



    #forward propagation
    def forward_prop(self, X):
        #calculate weighted sum, and then activation for each layer
        #input layer
        Z1 = self.W[0].dot(X) + self.b[0]
        A1 = self.ReLU(Z1)

        #hidden layer 1
        Z2 = self.W[1].dot(A1) + self.b[1]
        A2 = self.ReLU(Z2)

        #hidden layer 2
        Z3 = self.W[2].dot(A2) + self.b[2]
        A3 = self.softmax(Z3)

        return Z1, A1, Z2, A2, Z3, A3



    #backward propagation
    def back_prop(self, Z1, A1, Z2, A2, Z3, A3, X, Y):
        #get Y as matrix
        expected_Y = self.result_arr(Y)

        #derivative of the loss function, and loss with regards to weights and biases of the second hidden layer
        dZ3 = A3 - expected_Y
        dW3 = 1 / self.examples * dZ3.dot(A2.T)
        db3 = 1 / self.examples * np.sum(dZ3)

        #derivative of ReLu, and loss with regards to weights and biases of the first hidden layer
        dZ2 = self.W[2].T.dot(dZ3) * self.dReLU(Z2)
        dW2 = 1 / self.examples * dZ2.dot(A1.T)
        db2 = 1 / self.examples * np.sum(dZ2)

        #derivative of ReLu, and loss with regards to weights and biases of the input layer
        dZ1 = self.W[1].T.dot(dZ2) * self.dReLU(Z1)
        dW1 = 1 / self.examples * dZ1.dot(X.T)
        db1 = 1 / self.examples * np.sum(dZ1)

        return dW1, db1, dW2, db2, dW3, db3



    #simply updates weights and biases of the respective layers
    def update_params(self, dW1, db1, dW2, db2, dW3, db3, learning_rate):
        self.W[0] = self.W[0] - learning_rate * dW1
        self.b[0] = self.b[0] - learning_rate * db1
        self.W[1] = self.W[1] - learning_rate * dW2
        self.b[1] = self.b[1] - learning_rate * db2
        self.W[2] = self.W[2] - learning_rate * dW3
        self.b[2] = self.b[2] - learning_rate * db3



    #relu function, return Z if larger than 0, else return 0
    def ReLU(self, Z):
        return np.maximum(Z, 0)



    #derivative of relu, true = 1, false = 0
    #the slope is 1 if greater than 0, and 0 if less than 0
    def dReLU(self, Z):
        return Z > 0



    #softmax function (get the probability distribution for Z)
    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A



    #creates an array [sum(training examples), 10], where incorrect values are 0, and correct value is 1
    def result_arr(self, Y):
        #create array of zeros, x-size = the number of training examples, y-size =... Y is 0-9, so 9 + 1 = 10
        arr_Y = np.zeros((Y.size, Y.max() + 1))

        #basically loop through all rows of arr_Y, and set column Y to 1
        arr_Y[np.arange(Y.size), Y] = 1

        #transpose and return it
        arr_Y = arr_Y.T
        return arr_Y



    #train network by gradient descent method
    def train_network(self, X, Y, learning_rate, iterations):

        for i in range(iterations):
            #do forward propagation
            Z1, A1, Z2, A2, Z3, A3 = self.forward_prop(X)

            #do backward propagation
            dW1, db1, dW2, db2, dW3, db3 = self.back_prop(Z1, A1, Z2, A2, Z3, A3, X, Y)

            #update weights and biases
            self.update_params(dW1, db1, dW2, db2, dW3, db3, learning_rate)

            if i % 10 == 0:
                print("Iteration: ", i)
                print("Accuracy: ", self.get_accuracy(self.get_predictions(A3), Y))



    #get predicted number value
    def get_predictions(self, Y_prediction):
        return np.argmax(Y_prediction, 0)



    #calculate average accuracy of predictions
    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size


    
    #get predictions from the trained network on input X (can be matrix)
    def make_predictions(self, X):
        _, _, _, _, _, Y_prediction = self.forward_prop(X)
        predictions = self.get_predictions(Y_prediction)
        Y_acc = Y_prediction / np.sum(Y_prediction) * 100
        Y_acc = np.around(Y_acc)
        Y_acc = Y_acc.astype(int)
        Y_acc = list(Y_acc.flatten())
        Y_acc = [int(i) for i in Y_acc]
        return predictions, Y_acc



    #debug/test function
    def test_prediction(self, index):

        print(self.X_test[:, index, None])

        current_image = self.X_test[:, index, None]
        prediction = self.make_predictions(self.X_test[:, index, None])
        label = self.Y_test[index]
        print("Prediction: ", prediction)
        print("Label: ", label)
        
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()



    def run_tests(self):
        for i in range(1):
            self.test_prediction(i)



    #for getting and predicting a single digit
    def predict(self, data):
        data = [data] * 1
        data = np.array(data).T
        X = data[0:len(data)]
        X = X / 255.
        return self.make_predictions(X)
