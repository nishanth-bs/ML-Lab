import numpy as np
ipData = np.array([[2,9],[1,5],[3,6]], dtype=float) 
expectedOp=np.array ([[92], [86], [89]], dtype=float) 
ipData /= np.amax(ipData, axis=0)
expectedOp/=100 
def sigmoid(x):
    return 1/(1+np.exp(-x)) 
def derivative_sigmoid(x):
    return x*(1-x)
epoch=1
learning_rate = 0.5
iplayerNeurons = 2 
hiddenlayerNeurons=3 
oplayerNeurons = 1 
hiddenlayerWt=np.random.uniform(size=(iplayerNeurons,
                                            hiddenlayerNeurons)) 
hiddenlayer_bias = np.random.uniform(size=(1, hiddenlayerNeurons)) 
outputlayer_weights=np.random.uniform(size= (hiddenlayerNeurons,
                                             oplayerNeurons ))  
outputlayer_bias = np.random.uniform(size=(1, oplayerNeurons)) 
for i in range(epoch):
    hiddenlayer_input = np.dot(ipData, hiddenlayerWt) 
    hiddenlayer_input = hiddenlayer_input + hiddenlayer_bias 
    hiddenlayer_output = sigmoid(hiddenlayer_input) 
    
    outputlayer_input = np.dot(hiddenlayer_output, outputlayer_weights) 
    outputlayer_input = outputlayer_input + outputlayer_bias
    outputlayer_output=sigmoid(outputlayer_input)
     
    outputlayer_error=expectedOp-outputlayer_output 
    outputlayer_gradient=derivative_sigmoid(outputlayer_output) 
    outputlayer_error_correction = outputlayer_error* outputlayer_gradient
   
    hiddenlayer_error = outputlayer_error_correction.dot (outputlayer_weights. T) 
    hiddenlayer_gradient=derivative_sigmoid(hiddenlayer_output) 
    hiddenlayer_error_correction=hiddenlayer_error*hiddenlayer_gradient
    
    outputlayer_weights += hiddenlayer_output.T.dot(outputlayer_error_correction) * learning_rate 
    hiddenlayerWt += ipData.T.dot(hiddenlayer_error_correction)*learning_rate
    
print("Input : ", ipData)
print("Expected Output :", expectedOp)
print("actual output: ",outputlayer_output)
