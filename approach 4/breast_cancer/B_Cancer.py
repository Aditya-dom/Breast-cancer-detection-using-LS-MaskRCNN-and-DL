import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def __init__():
    train_x=pd.read_csv("cancer_data.csv")
    train_x = np.array(train_x)
    train_y=pd.read_csv("cancer_data_y.csv")
    train_y = np.array(train_y)
#0.00002-92
#0.000005-92, 93 when 200000

    d = model(train_x.T, train_y.T, num_iters=190500 , alpha=0.000000065, print_cost=True)

# Logistic Regression as deep learning



def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s


def initialize(dim):
    """

    :param dim: size of vector w initilazied with zeros
    :return:
    """
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w,   b


def propagate(w, b, X, Y):
    """

    :param w: weights for w
    :param b: bias
    :param X: size of data(no of features, no of examples)
    :param Y: true label
    :return:
    """
    m=X.shape[1] #getting no of rows


    #Forward Prop
    A=sigmoid(np.dot(w.T,X)+b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    #backwar prop
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)
    grads={"dw":dw,
           "db":db}
    return grads,cost

def optimize(w,b,X,Y,num_iters,alpha,print_cost=False):
    """

    :param w: weights for w
    :param b: bias
    :param X: size of data(no of features, no of examples)
    :param Y: true label
    :param num_iters: number of iterations for gradient
    :param alpha:
    :return:
    """

    costs=[]
    for i in range(num_iters):
        grads, cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        w=w-alpha*dw
        b=b-alpha*db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
                  "b": b}

    grads = {"dw": dw,
                 "db": db}

    return params, grads, costs

def predict(w,b,X):
    """

    :param w:
    :param b:
    :param X:
    :return:
    """
    m=X.shape[1]
    y_pred=np.zeros(shape=(1,m))
    w=w.reshape(X.shape[0],1)

    A=sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        y_pred[0,i]=1 if A[0,i]>0.5 else 0

    assert (y_pred.shape==(1,m))
    return y_pred


def model(X_train, Y_train, num_iters=2000, alpha=0.5, print_cost=False):
    """

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """

    w,b=initialize(X_train.shape[0])
    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_iters,alpha,print_cost)

    w=parameters["w"]
    b=parameters["b"]

    y_prediction_train=predict(w,b,X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - Y_train)) * 100))
    d = {"costs": costs,
         #"Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": alpha,
         "num_iterations": num_iters}

    # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()

    return d


__init__()

