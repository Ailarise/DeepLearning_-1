import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize']=(5.0,4.0)   #set default size of plots
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'


np.random.seed(1)

def initialize_parameters_deep(layer_dims):  #初始化参数

       np.random.seed(3)
       parameters={}
       L=len(layer_dims)

       for i in range(1,L):
              parameters['W'+str(i)]=np.random.randn(layer_dims[i], layer_dims[i-1])/np.sqrt(layer_dims[i-1])
              parameters['b'+str(i)]=np.zeros((layer_dims[i],1))

              assert (parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))  #检验错误
              assert (parameters['b' + str(i)].shape == (layer_dims[i], 1))

       return parameters


def linear_forward(A,W,b):  #前向传播

       Z=np.dot(W,A)+b

       assert (Z.shape==(W.shape[0], A.shape[1]))
       cache=(A,W,b)

       return Z,cache



def linear_activation_forward(A_prev,W,b,activation):  #激活函数

       if activation=="sigmoid":
              Z,linear_cache=linear_forward(A_prev,W,b)
              A,activation_cache=sigmoid(Z)

       elif activation=="relu":
              Z,linear_cache=linear_forward(A_prev,W,b)
              A, activation_cache = relu(Z)

       assert(A.shape==(W.shape[0],A_prev.shape[1]))
       cache=(linear_cache,activation_cache)

       return A,cache



def L_model_forward(X,parameters):  #网络前向传播模型

       caches=[]
       A=X
       L=len(parameters)//2

       for i in range(1,L):
              A_prev=A
              A, cache=linear_activation_forward(A_prev, parameters['W'+str(i)],parameters['b'+str(i)],activation="relu")
              caches.append(cache)

       AL,cache=linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation="sigmoid")
       caches.append(cache)

       assert(AL.shape==(1,X.shape[1]))

       return AL,caches



def compute_cost(AL, Y):   #计算代价函数

       m=Y.shape[1]

       cost=-1/m * np.sum(Y*np.log(AL)+(1-Y)*(np.log(1-AL)))

       cost=np.squeeze(cost)
       assert(cost.shape==())

       return cost



def linear_backward(dZ, cache):  #反向传播

       A_prev,W,b=cache
       m=A_prev.shape[1]

       dW=1/m*np.dot(dZ, A_prev.T)
       db=1/m*np.sum(dZ,axis=1,keepdims=True)
       dA_prev=np.dot(W.T,dZ)

       assert (dA_prev.shape == A_prev.shape)
       assert (dW.shape == W.shape)
       assert (db.shape == b.shape)

       return dA_prev, dW,db



def linear_activation_backward(dA, cache, activation):

       linear_cache, activation_cache=cache

       if activation=="relu":
              dZ=relu_backward(dA, activation_cache)
              dA_prev, dW, db = linear_backward(dZ, linear_cache)

       elif activation=="sigmoid":
              dZ=sigmoid_backward(dA, activation_cache)
              dA_prev, dW, db = linear_backward(dZ, linear_cache)

       return dA_prev, dW, db



def L_model_backward(AL,Y,caches):

       grads={}
       L=len(caches)
       m=AL.shape[1]
       Y=Y.reshape(AL.shape)   #Y与AL维度一致

       dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

       current_cache=caches[L-1]
       grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,'sigmoid')

       for l in reversed(range(L-1)):

              current_cache=caches[l]
              dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
              grads["dA" + str(l + 1)] = dA_prev_temp
              grads["dW" + str(l + 1)] = dW_temp
              grads["db" + str(l + 1)] = db_temp

       return grads


def update_parameters(parameters, grads, learning_rate):  #更新学习率

       L=len(parameters)//2

       for l in range(L):
              parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
              parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

       return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations=3000, print_cost=False):

       np.random.seed(1)
       costs=[]

       parameters=initialize_parameters_deep(layers_dims)

       for i in range(0,num_iterations):

              AL,caches=L_model_forward(X,parameters)
              cost=compute_cost(AL,Y)
              grads=L_model_backward(AL,Y,caches)
              parameters=update_parameters(parameters,grads,learning_rate)

              if print_cost and i % 100 == 0:
                     print("Cost after iteration %i: %f" % (i, cost))
              if print_cost and i % 100 == 0:
                     costs.append(cost)

       plt.plot(np.squeeze(costs))
       plt.ylabel('cost')
       plt.xlabel('iterations (per tens)')
       plt.title("Learning rate =" + str(learning_rate))
       plt.show()

       return parameters


def predict(X, y, parameters):
       """
       This function is used to predict the results of a  L-layer neural network.

       Arguments:
       X -- data set of examples you would like to label
       parameters -- parameters of the trained model

       Returns:
       p -- predictions for the given dataset X
       """

       m = X.shape[1]
       n = len(parameters) // 2  # number of layers in the neural network
       p = np.zeros((1, m))

       # Forward propagation
       probas, caches = L_model_forward(X, parameters)


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
index = 7
#plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]


# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

#layers_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_y, [12288, 20, 7, 5, 1], learning_rate=0.0075, num_iterations = 2500, print_cost = True)


pred_train = predict(train_x, train_y, parameters)

pred_test = predict(test_x, test_y, parameters)

print_mislabeled_images(classes, test_x, test_y, pred_test)





















