import numpy as np

class NeuralNetwork:
    # 简单的全连接神经网络
    def __init__(self, input_size=28*28, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.gradientsWeights = []
        self.gradientsBiases = []
        self.iterations = 0

        # 隐藏层
        self.weights.append(np.random.randn(input_size, hidden_layers[0]) * np.sqrt(2. / input_size))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        for i in range(len(hidden_layers)-1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i+1]))
            self.biases.append(np.zeros((1, hidden_layers[i+1])))

        self.weights.append(0.01 * np.random.randn(hidden_layers[len(hidden_layers)-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, inputs):
        #inputs = inputs.reshape(inputs.shape[0], -1)
        self.outputs = [inputs]
        self.outputsTesting = ["inputs"]

        for i in range(len(self.weights)):
            self.outputs.append(np.dot(self.outputs[-1], self.weights[i]) + self.biases[i])
            self.outputsTesting.append(["dense"])

            if i == len(self.weights)-1:
                finalOutput = np.exp(self.outputs[-1] - np.max(self.outputs[-1], axis=1, keepdims=True))
                finalOutput = finalOutput / np.sum(finalOutput, axis=1, keepdims=True)
                self.outputs.append(finalOutput)
                self.outputsTesting.append(["softmax"])
            else:
                self.outputs.append(np.maximum(0, self.outputs[-1]))
                self.outputsTesting.append(["ReLU"])
        
        return self.outputs[-1]

    def backwards(self, y_true):

        samples = len(self.outputs[-1])

        if len(y_true.shape) == 2:
            
            y_true = np.argmax(y_true, axis=1)

        dSoftMaxCrossEntropy = self.outputs[-1].copy()
        dSoftMaxCrossEntropy[range(samples), y_true] -= 1
        dSoftMaxCrossEntropy = dSoftMaxCrossEntropy / samples

        dInputs = np.dot(dSoftMaxCrossEntropy.copy(), self.weights[-1].T)

        dWeights = np.dot(self.outputs[-3].T, dSoftMaxCrossEntropy.copy())
        dBiases = np.sum(dSoftMaxCrossEntropy.copy(), axis=0, keepdims=True)

        self.gradientsBiases = [dBiases] + self.gradientsBiases
        self.gradientsWeights = [dWeights] + self.gradientsWeights

        i = -3
        j = -1
        for _ in range(len(self.hidden_layers)):
            i -= 1
            j -= 1
            dInputsReLU = dInputs.copy()
            dInputsReLU[self.outputs[i] <= 0] = 0

            i -= 1
            dInputs = np.dot(dInputsReLU, self.weights[j].T)
            dWeights = np.dot(self.outputs[i].T, dInputsReLU)
            dBiases = np.sum(dInputsReLU, axis=0, keepdims=True)
            self.gradientsWeights = [dWeights] + self.gradientsWeights
            self.gradientsBiases = [dBiases] + self.gradientsBiases


    def updateParams(self, lr, decay):
        lr = lr * (1. / (1. + decay * self.iterations))

        for i in range(len(self.weights)):
            #assert self.weights[i].shape == self.gradientsWeights[i].shape
            self.weights[i] += -lr*self.gradientsWeights[i]

        
        for i in range(len(self.biases)):
            #assert self.biases[i].shape == self.gradientsBiases[i].shape
            self.biases[i] += -lr*self.gradientsBiases[i]

        self.iterations+=1

def CategoryEntropy(yPred, yTrue):
    yPred = np.clip(yPred, 1e-10, 1-1e-10)

    loss = -np.sum(yTrue*np.log(yPred), axis=1)

    average_loss = np.mean(loss)

    return average_loss

def sparse_to_one_hot(sparse_labels, num_classes):
    one_hot_encoded = np.zeros((len(sparse_labels), num_classes))
    one_hot_encoded[np.arange(len(sparse_labels)), sparse_labels]=1
    return one_hot_encoded