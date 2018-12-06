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
    hiddenlayerIp = np.dot(ipData, hiddenlayerWt) 
    hiddenlayerIp = hiddenlayerIp + hiddenlayer_bias 
    hiddenlayerOp = sigmoid(hiddenlayerIp) 
    
    outputlayerIp = np.dot(hiddenlayerOp, outputlayer_weights) 
    outputlayerIp = outputlayerIp + outputlayer_bias
    outputlayerOp=sigmoid(outputlayerIp)
     
    outputlayerErr=expectedOp-outputlayerOp 
    outputlayerGradient=derivative_sigmoid(outputlayerOp) 
    outputlayerErr_correction = outputlayerErr* outputlayerGradient
   
    hiddenlayer_error = outputlayerErr_correction.dot (outputlayer_weights. T) 
    hiddenlayer_gradient=derivative_sigmoid(hiddenlayerOp) 
    hiddenlayer_error_correction=hiddenlayer_error*hiddenlayer_gradient
    
    outputlayer_weights += hiddenlayerOp.T.dot(outputlayerErr_correction) * learning_rate 
    hiddenlayerWt += ipData.T.dot(hiddenlayer_error_correction)*learning_rate
    
print("Input : ", ipData)
print("Expected Output :", expectedOp)
print("actual output: ",outputlayerOp)
