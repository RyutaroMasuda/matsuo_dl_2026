import numpy as np
# import matplotlib
# matplotlib.use("QTAgg")
import matplotlib.pyplot as plt
import ipdb

def relu(x):
    return np.maximum(0,x)
def deriv_relu(x):
    return (x>0).astype(x.dtype)
def sigmoid(x):
    return np.exp(np.maximum(0,x))/(1+np.exp(-np.abs(x)))
def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

x_train_xor = np.array([[0,1],[1,0],[0,0],[1,1]]) # (4,2)
t_train_xor = np.array([[1],[1],[0],[0]]) # (4,1)
x_valid_xor, t_valid_xor = x_train_xor, t_train_xor

plt.figure(figsize=(6,6))
plt.hlines([0], xmin=-1, xmax=2, color="black", alpha=0.7)
plt.vlines([0], ymin=-1, ymax=2, color="black", alpha=0.7)
plt.scatter(x_train_xor[0:2, 0], x_train_xor[0:2, 1], color="red", label="1")
plt.scatter(x_train_xor[2:, 0], x_train_xor[2:, 1], color="blue", label="0")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.xlim([-0.5, 1.5])
plt.ylim([-0.5, 1.5])
plt.legend()
plt.savefig("output.png")

W1 = np.random.uniform(low=-0.08,high=0.08,size=(2,8)).astype("float64")
b1 = np.zeros(8).astype("float64")
W2 = np.random.uniform(low=-0.08, high=0.08, size=(8,1)).astype("float64")
b2 = np.zeros(1).astype("float64")

def train_xor(x:np.ndarray,t:np.ndarray,eps:float) -> float:
    """
    :param x: input data, (batch_size, input_dim)
    :param t: ground-truth labels, (batcyh_size, output_dim)
    :param eps: learning rate
    :return cost : value function, cross-entropy 
    """
    global W1,b1,W2,b2
    batch_size = x.shape[0]

    # Forward Propagation
    u1 = np.matmul(x,W1) + b1
    h1 = relu(u1)
    u2 = np.matmul(h1,W2) + b2
    y = sigmoid(u2)

    # Compute loss
    cost = -(t*np.log(y) + (1-t)*np.log(1-y)).mean()

    # Back Propagation
    delta_2 = y-t # (batch_size, 1)
    delta_1 = deriv_relu(u1) * np.matmul(delta_2,W2.T) # (batch_size, 8)
    # ipdb.set_trace()
    dW1 = np.matmul(x.T,delta_1) / batch_size
    db1 = np.matmul(np.ones(batch_size), delta_1) / batch_size
    
    dW2 = np.matmul(h1.T,delta_2) / batch_size
    db2 = np.matmul(np.ones(batch_size), delta_2) / batch_size
    
    W1 -= eps * dW1
    b1 -= eps * db1
    W2 -= eps * dW2
    b2 -= eps * db2

    return cost

def valid_xor(x,t):
    global W1,b1,W2,b2
    
    u1 = np.matmul(x, W1) + b1
    h1 = relu(u1)
    u2 = np.matmul(h1, W2) + b2
    y = sigmoid(u2)

    cost = -(t*np.log(y)+(1-t)*np.log(1-y)).mean()

    return cost, y

for epoch in range(3000):
    for x,t in zip(x_train_xor,t_train_xor):    
        cost = train_xor(x[None,:],t[None,:],eps=0.05)
    cost, y_pred = valid_xor(x_valid_xor,t_valid_xor)
    print(f"epoch{epoch}:{y_pred}")