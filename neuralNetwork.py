import numpy as np

class InputLayer:

    def __init__(self, number_of_input): 
        self.number_of_input = number_of_input
    
    #to add the vlaues in the network 
    def use(self, input_matrix):
        if isinstance(input_matrix,np.ndarray):
            return input_matrix
        else:
            self.out_matrix = np.array(input_matrix)
            return self.out_matrix

class HiddenLayer:
    def __init__(self, number_of_input, number_of_neurons, activation='relu'):
        low, high = -1, 1
        self.weight_matrix = np.random.uniform(low, high, size=(number_of_input, number_of_neurons))
        self.bias = np.random.uniform(low, high, size=(1, number_of_neurons))
        self.activation = activation

    def apply_activation(self, values):
        if self.activation == 'relu':
            return np.maximum(0, values)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-values))
        elif self.activation == 'tanh':
            return np.tanh(values)
        else:
            return values  # No activation (identity)

    def forward(self, input_matrix):
        values = np.dot(input_matrix, self.weight_matrix) + self.bias
        return self.apply_activation(values)


class Model:
   
    def __init__(self):
        self.inputNumber = input('Enter the number of input paramentes') # input nodes
        self.numberOfLayers = 3 #number of hidden layer and output layer

        self.listOfHiddenLayers = []
        self.inLayerObj = InputLayer(self.inputNumber)

        for i in range(self.numberOfLayers):
            number_of_input = int(input('number of input: '))
            numberOfNeurons = int(input('number of neurons: '))
            activation = input('Enter the activation funtion: ')
            hidden_layer_obj = HiddenLayer(number_of_input,numberOfNeurons,activation)

            self.listOfHiddenLayers.append(hidden_layer_obj)
    
    #to get cost funtion of the layer 
    def costFuntion(self,value_matrix,req_arr,):
        
        cost_funtion = np.subtract(value_matrix,req_arr)**2

        return cost_funtion

    #pridict while training
    def train_pridict(self,features):
        
        arr = self.inLayerObj.use(features)

        for layer in self.listOfHiddenLayers:
            arr = layer.forward(arr)
        
        return arr
    
    def magic(self,value_matrix,label,layerSize):
        pass

    #to train the data
    def train(self,data):
        
        n = len(data[0]) - 2 

        for row in data:
            
            label = row[0]
            features = row[1:n]
        
        temp_arr = self.train_pridict(features)
        

    def pridict(self,data):

        arr = self.inLayerObj.use(data)
        
        for layer in self.listOfHiddenLayers:
            arr = layer.forward(arr)

        pridiction_value = -1
        pridiction = None

        for i,obj in enumerate(arr[0]):
            if pridiction_value < obj:
                pridiction = {'label' : i, 'confidence' : obj}
        
        return pridiction