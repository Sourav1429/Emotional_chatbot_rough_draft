import numpy as np
from math import exp,pow
from random import random,seed
from matplotlib import pyplot as plt
import prediction_try
from sklearn.linear_model import LinearRegression
from math import sqrt
from XNOR import CaseA
import sys
import pandas as pd
from check_inv_cont import MappingEmotion
def get_data(network,num):
        t=[];
        for i in range(num):
                data=network[i];
                for j in range(len(data)):
                        rown=data[j];
                        t.append(rown['cost_without_activation'])
        return np.array(t)
def data_analysis(x,per,y):
        for i in range(len(y)):
                print('>>expected:',y[i],'\t>>got:',x[i],'\t>>relation_dynamic:',per[i])
def test(y1,y2):
        t=[];
        if(len(y1) != len(y2)):
                print("Wrong dimensions. Dimensionality mis-match")
                sys.exit()
        for i in range(len(y1)):
                if(y1[i]==y2[i]):
                        t.append(1)
                else:
                        t.append(0)
        return t;
# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
	print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse

# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions

# Test simple linear regression
'''dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
rmse = evaluate_algorithm(dataset, simple_linear_regression)
print('RMSE: %.3f' % (rmse))'''
class Dynamics:
    ''' Defining the parameters of the class
    Inside this class, the functions relative to defining of weights and other parameters will be done.
    Steps for creation
    1) Defining the weights rather the network
    2) Perform sigmoid activation
    3) Conversion into a continuous model
    4) Defining of second network
    5) Feed forward path
    7) Back-prop training
    8) Previous data intake
    9) Mapping'''
    
    def weights(self,w1,w2,w3):
        return np.array([w1,w2,w3]);
    ''' conversion of list into array for defining our kernel'''
    def weighted_sum(self,row,w):
        return np.dot(w.T,row)
    #caluculating the weighted sum
    def sigmoid(self,Z):
        return 1/(1+exp(-Z))
    #function to find the sigmoid value
    def transfer_function(self,Z):
        s=self.sigmoid(Z);
        return s*(1-s);
    #The final transfer derivative function required for backprop
    def form_network(self,n,m=0):
        if(m==1):
                network=[[{'weights':[0.5,0.5,0]}]for i in range(n)]
                return network; 
        network=[[{'weights':[0.5,-0.5,0.3]}]for i in range(n)]
        return network;
    #formation of our network
    def u_step(self,Z,val=0.3):
        thr=val
        if(Z>=thr):
            return 1.0;
        else:
            return 0.001;
    #Defining the unit step function
    def feed_forward(self,network,data,val=0.3):
        Z=0;
        for i in range(len(data)):
            rowd=data[i]
            layer=network[i];
            for j in range(len(layer)):
                rown=layer[j]
                Z=rown['weights'][-1];
                for k in range(len(rown['weights'])-1):
                    Z=Z+rowd[k]*rown['weights'][k];
                rown['cost_without_activation']=Z;
                rown['cost_with_activation']=self.u_step(Z,val);
        return network;
    #Feed_forward training
    '''network formed. Order of dictionary is such that the first 2 weights are genuine weights and third one is bias'''
    def continuity(self,mean,sd,Z):
        x=Z-mean;
        x=x/(2*pow(sd,2));
        expr=exp(-x)/(sd*pow(2*np.pi,0.5));
    #Defining the Gaussian Distribution continuous function
    def plot(self,network,data):
        points=list();active=list();data_points=list();error=list();
        for i in range(len(network)):
            rowd=data[i]
            layer=network[i];
            data_points.append(rowd[-1])
            for j in range(len(layer)):
                rown=layer[j]
                points.append(rown['cost_without_activation'])
                active.append(rown['cost_with_activation'])
            ##        print(points)
            ##        print(len(points))
            ##        print(active)
            ##        print(len(active))
            ##        print(data_points)
            ##        print(len(data_points))
        for i in range(len(data_points)):
            error.append(abs(active[i]-data_points[i]))
        threshold=[0.3 for i in range(len(points))]
        '''Defining the subplots'''
        fig,axs=plt.subplots(2,2)
        axs[0][0].scatter(range(len(points)),points)
        axs[0][0].plot(range(len(points)),threshold)
        #axs[0][0].set_xlabel('Data value number');
        axs[0][0].set_ylabel('Cost without activation')
        axs[0][0].set_title('Scatter plot of data values before running activation');
        #plt.xlabel('Data value number from 0 to 9 for each graph')
        #first subplot is over heading over to the next one
        axs[0][1].scatter(range(len(active)),active)
        axs[0][1].plot(range(len(active)),threshold)
        #axs[0][1].set_xlabel('Data value number');
        axs[0][1].set_ylabel('Cost with activation')
        axs[0][1].set_title('Scatter plot of data values after running activation');
        #second plot over heading over to the third one
        axs[1][0].scatter(range(len(data_points)),active)
        axs[1][0].plot(range(len(active)),threshold)
        #axs[1][0].set_xlabel('Data value number');
        axs[1][0].set_ylabel('Expected output')
        axs[1][0].set_title('Scatter plot of data values expected output');
        #third plot over heading over to the next plot
        axs[1][1].scatter(range(len(error)),error)
        axs[1][1].plot(range(len(error)),error)
        #axs[1][1].set_xlabel('Data value number');
        axs[1][1].set_ylabel('Error')
        axs[1][1].set_title('Scatter plot of error');
        axs[1][1].set_ylim(-0.001,0.0015)
        #plt.title('Data value number from 0 to 9 for each graph')
        fig.text(0.5,0.04,'Data value number from 0 to 9 for each graph',va='center',ha='center')
        plt.show()
#Plotting over
''' =========================================================================================================================================
                                            End of class
=========================================================================================================================================='''
seed(1)        
s=Dynamics()
network=s.form_network(10)
#Defining the network complete
#Defining the data for each pair
data_A_B=[[0.74,0.26,1],
      [0.13,0.87,0],
      [0.65,0.35,1],
      [0.98,0.02,1],
      [0.05,0.95,0],
      [0.5,0.5,1],
      [0.6,0.4,1],
      [0.2,0.8,0],
      [0.02,0.98,0],
      [0.38,0.62,0]]
#A-B rel dynamics data complete
data_B_C=[];
for i in range(10):
        l=random()
        if(l>=0.3):
                f=1
        else:
                f=0
        data_B_C.append([l,1-l,f])
#B_C data complete
data_A_C=[]
for i in range(10):
        m=(data_A_B[i][0]+data_B_C[i][0])/2.0
        data_A_C.append([m,1-m])
'''-----------------------------------------------------------------------------
-----------------------Plotting user data---------------------------------------'''
#Moving on to create plots
n_data=[];
for i in data_A_B:
    n_data.append(i[0]);
t=prediction_try.cumulative(n_data);
n_data=np.array(t)
#Cumulative sum for data_A_B complete
n_data2=[];
for i in data_B_C:
    n_data2.append(i[0]);
ti=prediction_try.cumulative(n_data2);
n_data2=np.array(ti)
#Cumulative sum for data_B_C complete
time=np.reshape(np.array([*range(1,11)]),(10,1))
#Defining the time axis complete
network=s.feed_forward(network,data_A_B);
#s.plot(network,data_A_B)
#Plotting A_B data complete
plt.figure(2)
network=s.feed_forward(network,data_B_C);
s.plot(network,data_B_C)
#Plotting B_C data complete
y1=[i[2] for i in data_A_B]
y2=[i[2] for i in data_B_C]
testval=test(y1,y2);
#obtain the testing report
#Form the final training network
#Go for the feed forward operation
data=[[data_A_B[i][0],data_B_C[i][0]] for i in range(10)]
f_network=[]
s1=CaseA()
for i in range(len(data)):
        f_network.append(s1.network([data[i][0],data[i][1]]))
x=[f_network[i][0] for i in range(10)]
let=[f_network[i][2] for i in range(10)]
#Function to get data for analysis
data_analysis(x,let,testval);
regressor=LinearRegression()
regressor.fit(time,n_data)
l=regressor.predict([[11]])
#print(l)
t.append(l[0])
plt.figure(2)
prediction_try.plotting(t,12);
new_A_B=l[0]-n_data[-1]
print("predicted friendliness value of A-B=",new_A_B)
print("After passing through unit step activation:",s.u_step(l[0]-n_data[-1]));
#Prediction and plotting for A-B data
regressor.fit(time,n_data2)
l=regressor.predict([[11]])
print(l)
ti.append(l[0])
plt.figure(4)
prediction_try.plotting(ti,12);
new_B_C=l[0]-n_data2[-1]
print("predicted friendliness value of B-C=",new_B_C)
print("After passing through unit step activation:",s.u_step(l[0]-n_data2[-1]));
#Prediction and plotting for B-C data
mean=(new_A_B+new_B_C)/2.0;
data_A_C.append([mean,1-mean])
#Calulating the next emotion parameters
print("The next emotion parameters are as follow:")
mp=MappingEmotion()
next_emotion_measurement=mp.inv_continuous(data_A_B[-1][0],data_A_B[-2][0],data_A_B[-3][0])
#print(data_A_C[-1][0],data_A_C[-2][0],data_A_C[-3][0])
print(next_emotion_measurement)
