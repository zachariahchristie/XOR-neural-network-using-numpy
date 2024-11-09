import numpy as np

#Maps any real number between 0 and 1.
#sigmoid(x) = (1) / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Find derivative of sigmoid
#sigmoid'(x) = sigmoid(x)(1 - sigmoid(x))
#Used to update weights in backwards pass
def sigmoid_derivative(x):
    return x * (1-x)

class NeuralNetwork:
    #initialises the neural network, constructor
    #Initialise the weights randomly to begin with
    def __init__(self, input_size, hidden_size, output_size):
        #Initialises weight matrix between input and hidden layers
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        #Initialises weight matrix between hidden and output layers
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
    
    #Our forward pass method, performs forward propagation
    def forward(self, X):
        #First performs dot product of X and weight matrix for hidden layer, then applies sigmoid function
        self.hidden = sigmoid(np.dot(X, self.weights_input_hidden))
        #First performs dot product of hidden layer activations and weight matrix for output layer, then applies sigmoid function
        self.output = sigmoid(np.dot(self.hidden, self.weights_hidden_output))
        return self.output
    
    #Our backwards pass method, performs back propogation
    #Computates gradients and applies them to the weights, reducing error in next forward pass
    def backward(self, X, y, learning_rate):
        #Calculates output error
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)
        
        #Calculates hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        #Updates weights based on self.hidden, output delta, and learning rate
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

    #Trains our network over number of epochs
    #Each epoch loop performs a forwards and backwards pass
    # Every 100 epochs we calculate mean squared error to show model is learning    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch & 100 == 0:
                loss = np.mean(np.square(y - self.output))
                print(f"Epoch {epoch} - Loss: {loss:.4f}")



#Sample data, where X is a 4 x 2 matrix representing a binary input for a XOR problem
#y is the target output for each row of X, goal is for neural network to learn how XOR works
X = np.array([[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

nn = NeuralNetwork(input_size = 2, hidden_size = 2, output_size = 1)
nn.train(X, y, epochs = 10000, learning_rate = 2)

print("Predictions after training:")
print(nn.forward(X))