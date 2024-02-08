import numpy as np

#input layer
class inputLayer:

    def __init__(self, number_of_input): 
        self.number_of_input = number_of_input
    
    #to add the vlaues in the network 
    def use(self, input_matrix):
        if isinstance(input_matrix,np.ndarray):
            self.out_matrix = input_matrix
            return input_matrix
        else:
            self.out_matrix = np.array(input_matrix)
            return self.out_matrix
    
    def fetchInputValues(self):
        return self.out_matrix

#hidden and output layers
class layers:
    
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
        print(f'shapeofIN = {np.shape(input_matrix)}, shapeofWI = {np.shape(self.weight_matrix)}') #debugging
        values = np.dot(input_matrix,self.weight_matrix) + self.bias
        self.neuronValues =  self.apply_activation(values)
        # print(self.neuronValues) #debugging
        return self.neuronValues
    
class Model:

    #to intiallize the model
    def __init__(self,):
        inputNumber = int(input('enter the input parameters: ')) # input nodes
        self.numberOfLayers = int(input('enter the number of layers: ')) #number of hidden layer and output layer

        self.listOfHiddenLayers = []
        self.inLayerObj = inputLayer(inputNumber)

        for i in range(self.numberOfLayers):
            number_of_input = int(input('number of input: '))
            numberOfNeurons = int(input('number of neurons: '))
            activation = input('Enter the activation funtion: ')
            hidden_layer_obj = layers(number_of_input,numberOfNeurons,activation)

            self.listOfHiddenLayers.append(hidden_layer_obj)
            
            if i == self.numberOfLayers-1:
                self.numberOfLabels = numberOfNeurons

    def pridict(self,data,ForTraining = False):

        arr = data
        
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

        pridiction = pridiction - reqArr
        i = self.numberOfLayers - 1

        layers = self.listOfHiddenLayers[i]
        gradient = np.multiply(pridiction,layers.compute_derivative(pridiction))
        prevNuronValues = self.listOfHiddenLayers[i-1].neuronValues
        pr = layers.weight_matrix.T - learningRate*(np.dot(gradient.T,prevNuronValues))
        layers.weight_matrix = pr.T
        layers.bias = layers.bias - learningRate*(gradient)
        # print(layers.bias) #debugging
        i -= 1

        while i:
            #to get gradient 
            layers = self.listOfHiddenLayers[i]
            pre_w = self.listOfHiddenLayers[i+1].weight_matrix
            newValues = layers.neuronValues
            # print(f'newValues = {i} - {newValues}') #debugging
            derValues = layers.compute_derivative(newValues)
            cur_gradient = np.multiply(derValues.T,pre_w)
            pr = np.dot(gradient,cur_gradient.T)
            prevNuronValues = self.listOfHiddenLayers[i-1].neuronValues
            
            #updating weights
            weights = layers.weight_matrix
            upWeights = weights.T - learningRate*(np.dot(pr.T,prevNuronValues))
            layers.weight_matrix = upWeights
            gradient = pr
            
            #updating the bias 
            layers.bias = layers.bias - learningRate*(pr)
            # print(layers.bias) #debugging

            i -= 1

        #for input layer - first layer value
        layer = self.listOfHiddenLayers[0]
        pre_w = self.listOfHiddenLayers[1].weight_matrix
        newValues = layers.neuronValue
        derValues = layers.compute_derivative(newValues)
        cur_gradient = np.dot(pre_w,gradient.T)
        pr = np.multiply(derValues,cur_gradient.T)
        prevNuronValues = data
        weights = layer.weight_matrix
        upWeights = weights.T - learningRate*(np.dot(pr.T,prevNuronValues))
        layers.weight_matrix = upWeights.T
        layers.bias = layers.bias - learningRate*(pr)
    
class ModelTest:

    #to intiallize the model
    def __init__(self,parameters):
        ind = 0
        inputNumber = parameters[ind] # input nodes
        ind += 1
        self.numberOfLayers = parameters[ind] #number of hidden layer and output layer
        ind += 1
        self.listOfHiddenLayers = []
        self.inLayerObj = inputLayer(inputNumber)

        for i in range(self.numberOfLayers):
            number_of_input = parameters[ind]
            ind += 1 
            numberOfNeurons = parameters[ind]
            ind += 1
            activation = parameters[ind]
            ind += 1
            hidden_layer_obj = layers(number_of_input,numberOfNeurons,activation)

            self.listOfHiddenLayers.append(hidden_layer_obj)
            
            if i == self.numberOfLayers-1:
                self.numberOfLabels = numberOfNeurons

    def pridict(self,data,ForTraining = False):

        arr = data
        
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
        print('here') #debugging
        pridiction = self.pridict(data,ForTraining = True)
        reqArr =np.array([0 for i in range(self.numberOfLabels)])
        reqArr[label] = 1
        costFuntion = sum((pridiction - reqArr)**2)

        pridiction = pridiction - reqArr
        i = self.numberOfLayers - 1

        layers = self.listOfHiddenLayers[i]
        gradient = np.multiply(pridiction,layers.compute_derivative(pridiction))
        prevNuronValues = self.listOfHiddenLayers[i-1].neuronValues
        pr = layers.weight_matrix.T - learningRate*(np.dot(gradient.T,prevNuronValues))
        layers.weight_matrix = pr.T
        layers.bias = layers.bias - learningRate*(gradient)
        # print(layers.bias) #debugging
        i -= 1

        while i:
            
            #to get gradient 
            layers = self.listOfHiddenLayers[i]
            pre_w = self.listOfHiddenLayers[i+1].weight_matrix
            newValues = layers.neuronValues
            # print(f'newValues = {i} - {newValues}') #debugging
            derValues = layers.compute_derivative(newValues)
            cur_gradient = np.multiply(derValues.T,pre_w)
            pr = np.dot(gradient,cur_gradient.T)
            prevNuronValues = self.listOfHiddenLayers[i-1].neuronValues
            
            #updating weights
            weights = layers.weight_matrix
            upWeights = weights.T - learningRate*(np.dot(pr.T,prevNuronValues))
            print(f'wei = {np.shape(upWeights)}') #debugging
            layers.weight_matrix = upWeights
            gradient = pr
            
            #updating the bias 
            layers.bias = layers.bias - learningRate*(pr)
            # print(layers.bias) #debugging

            i -= 1

        #for input layer - first layer value
        layer = self.listOfHiddenLayers[0]
        pre_w = self.listOfHiddenLayers[1].weight_matrix
        newValues = self.listOfHiddenLayers[1].neuronValues
        print(f'new = {np.shape(newValues)}') #debugging
        derValues = layers.compute_derivative(newValues)
        cur_gradient = np.dot(pre_w,gradient.T)
        pr = np.multiply(derValues,cur_gradient.T)
        prevNuronValues = data
        weights = layer.weight_matrix
        upWeights = weights.T - learningRate*(np.dot(pr.T,prevNuronValues))
        layers.weight_matrix = upWeights.T
        layers.bias = layers.bias - learningRate*(pr)