import numpy as np

class layersT:
    
    def __init__(self, number_of_input, number_of_neurons,activation='relu',wig=None,bia=None):
        low, high = -1, 1
        self.weight_matrix = wig
        self.bias = bia
        self.activation = activation

    def apply_activation(self, values):
        if self.activation == 'relu':
            return np.maximum(0, values)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-values))
        elif self.activation == 'tanh':
            return np.tanh(values)
        elif self.activation == 'softmax':
            e_x = np.exp(values - np.max(values, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)
        else:
            return values  # No activation (identity)

    def compute_derivative(self, values):
        if self.activation == 'relu':
            return (values > 0).astype(float)
        elif self.activation == 'sigmoid':
            sigmoid_output = self.apply_activation(values)
            return sigmoid_output * (1 - sigmoid_output)
        elif self.activation == 'tanh':
            tanh_output = self.apply_activation(values)
            return 1 - tanh_output**2
        elif self.activation == 'softmax':
            softmax_output = self.apply_activation(values)
            return softmax_output * (1 - softmax_output)  # Note: This derivative might not be used often for softmax
        else:
            return np.ones_like(values)  # Derivative of the identity function is 1

    def forward(self, input_matrix):
        values = np.dot(input_matrix, self.weight_matrix) + self.bias
        self.neuronValues =  self.apply_activation(values)
        #print(f'r =  {self.neuronValues} values = {values}') #debugging
        return self.neuronValues
    
wig1 = np.array([[-0.80377147, -0.63344107],
                 [0.43278832, -0.02537794],
                 [-0.71144554, 0.79602224]])
wig2 = np.array([[ 0.30313998,-0.651056],
                 [ 0.11990958, -0.68028781]])
wig3 = np.array([[-0.13948543, -0.15244189 ,0.02360036],
                 [ 0.15260582 ,-0.41243941 ,0.62087499]])

b1 = np.array([-0.20960194,-0.86772309])
b2 = np.array([-0.04218223, 0.71097538])
b3 = np.array([-0.45757658, -0.6612067, 0.39908153])

testInputs = [3,2,'relu',wig1,b1,2,2,'relu',wig2,b2,2,3,'softmax',wig3,b3]

class ModelTest:

    #to intiallize the model
    def __init__(self) -> None:
        
        inputNumber = 3 # input nodes
        self.numberOfLayers = 3 #number of hidden layer and output layer

        self.listOfHiddenLayers = []
        self.inLayerObj = inputLayer(inputNumber)
        ind = 0
        for i in range(self.numberOfLayers):
            number_of_input = testInputs[ind]
            ind += 1
            numberOfNeurons = testInputs[ind]
            ind += 1
            activation = testInputs[ind]
            ind += 1
            w = testInputs[ind]
            ind += 1
            b = testInputs[ind]
            ind += 1
            hidden_layer_obj = layersT(number_of_input,numberOfNeurons,activation,w,b)

            self.listOfHiddenLayers.append(hidden_layer_obj)
            
            if i == self.numberOfLayers-1:
                self.numberOfLabels = numberOfNeurons

    def pridict(self,data,ForTraining = False):

        arr = self.inLayerObj.use(data)
        
        for layer in self.listOfHiddenLayers:
            arr = layer.forward(arr)

        if ForTraining:
            return arr

        pridiction_value = -1
        pridiction = None

        for i,obj in enumerate(arr[0]):
            if pridiction_value < obj:
                pridiction = {'label' : i, 'confidence' : obj}
        
        return pridiction
    

    def view_weights(self):
        for layer in self.listOfHiddenLayers:
            print(f'weights \n - {layer.weight_matrix}')
            print(f'bias - {layer.bias}')

    #backpropagation 
    def train(self,data,label,learningRate=0.1):
        
        pridiction = self.pridict(data,ForTraining = True)
        reqArr =np.array([0 for i in range(self.numberOfLabels)])
        reqArr[label] = 1
        costFuntion = sum((pridiction - reqArr)**2)
        
        # print(reqArr)
        # print(costFuntion)
        # print(f"prid - {pridiction}")

        pridiction = pridiction - reqArr
        i = self.numberOfLayers - 1

        while i:
            print(i)
            layers = self.listOfHiddenLayers[i]
            gradient = np.array([np.multiply(pridiction,layers.compute_derivative(pridiction))])
            print(gradient)
            a = np.array([self.listOfHiddenLayers[i-1].neuronValues])
            print(a)
            pr = layers.weight_matrix.T - learningRate*(np.dot(gradient.T,a))
            self.listOfHiddenLayers[i].weight_matrix = pr.T
            i -= 1

        layers = self.listOfHiddenLayers[0]
        gradient = np.array([np.multiply(pridiction,layers.compute_derivative(pridiction))])
        
        a = np.array([data])
        pr = layers.weight_matrix.T - learningRate*(np.dot(gradient.T,a))
        self.listOfHiddenLayers[0].weight_matrix = pr.T

