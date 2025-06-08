import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
   return 1/(1+np.exp(-x))

def forward_propogation(inputs,weights,biases):
    z1=np.dot(inputs,weights['w1'])+biases['b1']
    a1=sigmoid(z1)
    z2=np.dot(a1,weights['w2'])+biases['b2']
    a2=sigmoid(z2)
    return a2

#Example data
x= np.array([
   [5,8],
   [2,4],
   [8,7],
   [1,3]
])
y=np.array([[1],[0],[1],[0]])


# nueral network architecture
input_size=2
hidden_size=3
output_size=1

np.random.seed(42)
weights={
    'w1':np.random.rand(input_size,hidden_size),
    'w2':np.random.rand(hidden_size,output_size)
}
biases={
    'b1':np.zeros((1,hidden_size)),
    'b2':np.zeros((1,output_size))

}
#forward propogation
predictions= forward_propogation(x,weights,biases)

#plotting
plt.scatter(x[:,0],x[:,1],c=predictions[:,0],cmap='viridis',edgecolors='k',marker='o')
plt.title('Nueral Network Decision Boundary')
plt.xlabel('Study hours')
plt.ylabel('sleep hours')
plt.colorbar(label="Predicted Probablity of Passing")
plt.show()
