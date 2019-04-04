import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\STAR\Desktop\PYTHON\googleplaystore1.csv')
data.to_csv('googleplyatore1.csv',index=False)
#data = pd.read_csv(r"googleplaystore1.csv")
data.head()
data.info()
M=data[data.Installs == '0']
F=data[data.Installs != '0']

plt.scatter(M.Reviews,M.Rating)
plt.scatter(F.Reviews,F.Rating)
plt.show()

data.head(10841)
print(data.Type.value_counts())
data.drop(["Category"], axis=1,inplace=True)
data.drop(["App"], axis=1,inplace=True)
data.drop(["Size"], axis=1,inplace=True)
data.drop(["Price"], axis=1,inplace=True)
data.drop(["Content Rating"], axis=1,inplace=True)
data.drop(["Genres"], axis=1,inplace=True)
data.drop(["Last Updated"], axis=1,inplace=True)
data.drop(["Android Ver"], axis=1,inplace=True)
data.drop(["Current Ver"], axis=1,inplace=True)
data.set_index(["Reviews"],inplace=True)
#data.reset_index(drop=True)
#data.drop([" "], axis=1,inplace=True)
data.Installs=[1 if each == '0' else 0 for each in data.Installs]
print(data.info())

#Declaring Y
y=data.Installs.values

#Declaring X
x_data=data.drop(["Installs"], axis=1)
print(np.min(x_data))
#print(x_data)
 
#normalization
x = (x_data-np.min(x_data)) / (np.max(x_data)-np.min(x_data)).values
print(x)

#Train-test Split

x_train,x_test,y_train,y_test = train_test_split ( x,y, test_size=0.2,random_state=42)

x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T

data.loc[data['Installs'] == 0,'Installs'] = 0
data.loc[data['Installs'] != 0,'Installs'] = 1
columns = ['Installs','Rating']
train,cross_validation = train_test_split(data,test_size=0.2)
x_train = train[columns]
y_train = train['Type']
x_cross_validation = cross_validation[columns]
y_cross_validation = cross_validation['Type']

logst = LogisticRegression()
logst.fit(x_train,y_train)
print(logst.score(x_cross_validation,y_cross_validation)*100)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print("x_test", x_test.shape)
print("y_test", y_test.shape)

#1
# implementing initializing parameters and sigmoid function
def initialize_weights_and_bias (dimension):
    w=np.full((dimension,1),0.01)
    b=0.0
    return w,b
#w,b=initialize_weights_and_bias (3)

def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head
#sigmoid(0)

# implementing forward and backward propagation
def forward_backward_propagation(w,b,x_train,y_train):
    #forward propogation
    z=np.dot(w.T,x_train) + b
    y_head=sigmoid(z)
#    print(y_train.type())
#    y_train.to_frame().T
#    y_train.drop(["Reviews"], axis=1)
#    print(np.log(y_head))
#    print(np.dot(y_train,np.log(y_head)))
    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1]    #x_train.shape[1] for scaling
#backward propogation
    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]   #x_train.shape[1] for scaling
    derivative_bias=np.sum(y_head-y_train) / x_train.shape[1]  #x_train.shape[1] for scaling
    gradients={"derivative_weight": derivative_weight , "derivative_bias":derivative_bias}
    return cost,gradients

# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
#        cost_list.append(cost)
#        # lets update
#        w = w - learning_rate * gradients["derivative_weight"]
#        b = b - learning_rate * gradients["derivative_bias"]
#        if i % 10 == 0:
#            cost_list2.append(cost)
#            index.append(i)
#            print ("Cost after iteration %i: %f" %(i, cost))
#    # we update(learn) parameters weights and bias
#    parameters = {"weight": w,"bias": b}
#    plt.plot(index,cost_list2)
#    plt.xticks(index,rotation='vertical')
#    plt.xlabel("Number of Iterarion")
#    plt.ylabel("Cost")
#    plt.show()
#    return parameters, gradients, cost_list
#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)

# implementing prediction
def predict (w,b,x_test):  #x_test is a input for forward propagation
    z=sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction=np.zeros((1,x_test.shape[1]))
    for i in range (z.shape[1]):
        if z[0,i] <= 0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    return Y_prediction
#Logistic Regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
#    
#    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
#
#    # Print test Errors
#    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)