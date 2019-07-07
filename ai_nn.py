import numpy as np

class ActivationFunction:
    def __init__(self, func,dfunc):
        self.func = func
        self.dfunc = dfunc

sigmoid = ActivationFunction(
    lambda x:1 / (1 + np.exp(-x)),
    lambda y:y * (1 - y)
)
linear = ActivationFunction(
    lambda x:x,
    lambda y:1
)


tanh = ActivationFunction(
    lambda x:np.tanh(x),
    lambda y:1 - (y * y)
)

relu = ActivationFunction(
    lambda x:x * (x > 0),
    lambda y:1. * (y > 0)
)


class NeuralNetwork:
    def __init__(self,i_nodes, h_nodes, o_nodes, open = False):
        self.risetime_max = 0
        self.risetime_min = 0

        self.overshoot_max = 0
        self.overshoot_min = 0

        self.settling_max = 0
        self.settling_min = 0

        self.peak_max = 0
        self.peak_min = 0

        self.steady_max = 0
        self.steady_min = 0

        if isinstance(i_nodes, NeuralNetwork):
            temp = i_nodes

            self.input_nodes = temp.input_nodes
            self.hidden_nodes = temp.hidden_nodes
            self.output_nodes = temp.output_nodes

            self.weights_ih = temp.weights_ih.copy()
            self.weights_ho = temp.weights_ho.copy()

            self.bias_h = temp.bias_h.copy()
            self.bias_o = temp.bias_o.copy()
        else:
            self.input_nodes = i_nodes
            self.hidden_nodes = h_nodes
            self.output_nodes = o_nodes 
            
            if open == False:
                random_func = lambda x:x*2-1
                
                self.weights_ih = np.random.rand(self.input_nodes, self.hidden_nodes)
                func_ih = np.vectorize(random_func)
                self.weights_ih = np.matrix(func_ih(self.weights_ih))

                self.weights_ho = np.random.rand(self.hidden_nodes, self.output_nodes)

                func_ho = np.vectorize(random_func)
                self.weights_ho = np.matrix(func_ho(self.weights_ho))


                self.bias_h = np.random.rand(1, self.hidden_nodes)
                func_bias_h = np.vectorize(random_func)
                self.bias_h = np.matrix(func_bias_h(self.bias_h))

                self.bias_o = np.random.rand(1, self.output_nodes)
                func_bias_o = np.vectorize(random_func)
                self.bias_o = np.matrix(func_bias_o(self.bias_o))
                

        self.setActivation("sigmoid")
        self.setLearningRate(0.1)


    def setActivation(self, func):
        if func == "sigmoid":
            self.activation_function = sigmoid
        elif func == "tanh":
            self.activation_function = tanh
        elif func == "relu":
            self.activation_function = relu
        elif func == "linear":
            self.activation_function = linear
        self.activation_function_o = linear    
    
    def save(self):
        np.save("weights_ih", self.weights_ih)
        np.save("weights_ho", self.weights_ho)
        np.save("bias_h", self.bias_h)
        np.save("bias_o", self.bias_o)

        np.save("rt", ([self.risetime_min,self.risetime_max]))
        np.save("os", ([self.overshoot_min,self.overshoot_max]))
        np.save("st", ([self.settling_min,self.settling_max]))
        np.save("pk", ([self.peak_min,self.peak_max]))
        np.save("se", ([self.steady_min,self.steady_max]))

    def normalize(self, x):
        temp = []
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        x5 = []
        for i in range(len(x)-1):
            flag = False
            if i == 0:
                temp.append(x[0])
            else:
                for j in range(len(temp)):
                    if(x[i][0] == temp[j][0] and x[i][1] == temp[j][1] and x[i][2] == temp[j][2]):
                        flag = True
            if flag == False:
                temp.append(x[i])
                x1.append(float(x[i][4]))
                x2.append(float(x[i][5]))
                x3.append(float(x[i][6]))
                x4.append(float(x[i][7]))
                x5.append(float(x[i][8]))
            

        self.risetime_max = np.amax(x1,axis=0)
        self.risetime_min = np.amin(x1,axis=0)

        self.overshoot_max = np.amax(x2,axis=0)
        self.overshoot_min = np.amin(x2,axis=0)

        self.settling_max = np.amax(x3,axis=0)
        self.settling_min = np.amin(x3,axis=0)

        self.peak_max = np.amax(x4,axis=0)
        self.peak_min = np.amin(x4,axis=0)

        self.steady_max = np.amax(x5,axis=0)
        self.steady_min = np.amin(x5,axis=0)

        return temp

    def setLearningRate(self, r):
        self.learning_rate = r

    def funcMap(self, input, func):
        temp = input.copy()
        val = 0
        for i in range (input.shape[0]):
            for j in range (input.shape[1]):
                val = input[i,j]
                temp[i,j] = func(val)
        return temp
                
    def predict_from_model(self, input):
        self.weights_ih = np.load("weights_ih.npy")
        self.weights_ho = np.load("weights_ho.npy")

        self.bias_h = np.load("bias_h.npy")
        self.bias_o = np.load("bias_o.npy")

        rt = np.load("rt.npy")
        os = np.load("os.npy")
        st = np.load("st.npy")
        pk = np.load("pk.npy")
        se = np.load("se.npy")

        temp = input
        temp[0] = np.interp(input[0],rt,[0,1])
        temp[1] = np.interp(input[1],os,[0,1])
        temp[2] = np.interp(input[2],st,[0,1])
        temp[3] = np.interp(input[3],pk,[0,1])
        temp[4] = np.interp(input[4],se,[0,1])

        input = np.matrix(temp)

        hidden = np.dot(input, self.weights_ih)
        hidden = np.add(hidden, self.bias_h)

        hidden = self.funcMap(hidden, self.activation_function.func)

        output = np.dot(hidden, self.weights_ho)
        output = np.add(output, self.bias_o)

          
        output = self.funcMap(output, self.activation_function_o.func)
        
        return output

    def predict(self, input):
        temp = input
        temp[0] = np.interp(input[0],[self.risetime_min,self.risetime_max],[0,1])
        temp[1] = np.interp(input[1],[self.overshoot_min,self.overshoot_max],[0,1])
        temp[2] = np.interp(input[2],[self.settling_min,self.settling_max],[0,1])
        temp[3] = np.interp(input[3],[self.peak_min,self.peak_max],[0,1])
        temp[4] = np.interp(input[4],[self.steady_min,self.steady_max],[0,1])

        input = np.matrix(temp)

        hidden = np.dot(input, self.weights_ih)
        hidden = np.add(hidden, self.bias_h)

        hidden = self.funcMap(hidden, self.activation_function.func)

        output = np.dot(hidden, self.weights_ho)
        output = np.add(output, self.bias_o)

          
        output = self.funcMap(output, self.activation_function_o.func)
        
        return output
    
    def train(self, input, target):
        temp = input
        temp[0] = np.interp(input[0],[self.risetime_min,self.risetime_max],[0,1])
        temp[1] = np.interp(input[1],[self.overshoot_min,self.overshoot_max],[0,1])
        temp[2] = np.interp(input[2],[self.settling_min,self.settling_max],[0,1])
        temp[3] = np.interp(input[3],[self.peak_min,self.peak_max],[0,1])
        temp[4] = np.interp(input[4],[self.steady_min,self.steady_max],[0,1])

        input = np.matrix(temp)

        #Generate hidden input calculation, and add with bias
        hidden = np.dot(input, self.weights_ih)
        hidden = np.add(hidden, self.bias_h)

        #Activate hidden node
        hidden = self.funcMap(hidden, self.activation_function.func)

        #Generate Output of output node, and add with bias
        output = np.dot(hidden, self.weights_ho)
        output = np.add(output, self.bias_o)

        #Activate output node
        output = self.funcMap(output,  self.activation_function_o.func)

        #Calculate error of system prediction
        output_error = np.subtract(target, output)
        #print output_error[0, 0]

        #Calculate the gradien of output
        gradient = self.funcMap(output, self.activation_function_o.dfunc)
        gradient = np.multiply(gradient, output_error)
        gradient = np.multiply(gradient, self.learning_rate)

        #Calculate deltas
        hidden_t = np.transpose(hidden)
        weight_ho_delta = np.dot(hidden_t,gradient)
        
        #Adjust weight by the deltas
        self.weights_ho = np.add(self.weights_ho, weight_ho_delta)

        #Adjust bias by the gradien
        self.bias_o = np.add(self.bias_o, gradient)

        #Calculate hidden layer error
        who_t = np.transpose(self.weights_ho)
        hidden_error = np.dot(output_error, who_t)

        #Calculate hidden gradien
        hidden_gradien = self.funcMap(hidden, self.activation_function.dfunc)
        hidden_gradien = np.multiply(hidden_gradien, hidden_error)
        hidden_gradien = np.multiply(hidden_gradien, self.learning_rate)

        #calculate input node to hidden node deltas
        input_T = np.transpose(input)
        weight_ih_delta = np.dot(input_T, hidden_gradien)

        self.weights_ih = np.add(self.weights_ih, weight_ih_delta)
        self.bias_h = np.add(self.bias_h, hidden_gradien) 