import numpy as np
import matplotlib.pyplot as plt
#step1 design architcture
input_size=2
hidden_size=3
output_size=1
learning_rate=0.01
epochs=1000

#step2 initialization
np.random.seed(42)
weights_input_hidden=np.random.rand(input_size,hidden_size)
biases_hidden=np.zeros((1,hidden_size))
weights_hidden_output=np.random.rand(hidden_size,output_size)
biases_output=np.zeros((1,output_size))


#step3 activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#step4 mean square error
def mse_loss(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

#step5 training data
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

#training the neural network using back propogation
losses=[]

for epoch in range(epochs):
    #forward pass
    hidden_input=np.dot(x,weights_input_hidden)+biases_hidden
    hidden_output=sigmoid(hidden_input)
    output_layer_input=np.dot(hidden_output,weights_hidden_output)+biases_output
    predicted_output=sigmoid(output_layer_input)   

#calculate loss
loss = mse_loss(y,predicted_output)
losses.append(loss)

#..............backward pass
output_error = y-predicted_output
output_delta=output_error*sigmoid(predicted_output)*(1-sigmoid(predicted_output))

hidden_layer_error=output_delta.dot(weights_hidden_output.T)
hidden_layer_delta=hidden_layer_error*sigmoid(hidden_output)*(1-sigmoid(hidden_output))

#update weights and biases
weights_hidden_output += hidden_output.T.dot(output_delta)*learning_rate
biases_output += np.sum(output_delta, axis=0, keepdims=True)*learning_rate

weights_input_hidden += x.T.dot(hidden_layer_delta)*learning_rate
biases_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True)*learning_rate

#plot the loss curve
plt.plot(range(epochs),losses)
plt.xlabel('Epochs')
plt.ylabel('Mean Square Error loss')
plt.title('Training loss over Epochs')
plt.show()

#Make predictions after training
test_data=np.array([[0,0],[0,1],[1,0],[1,1]])
predictions=sigmoid(np.dot(sigmoid(np.dot(test_data,weights_input_hidden)+biases_hidden),weights_hidden_output)+biases_output)
print("Prediction after training:")
print(predictions)