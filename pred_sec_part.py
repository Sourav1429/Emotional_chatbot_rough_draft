import numpy as np
from math import exp,pow
from random import random,seed
from matplotlib import pyplot as plt
import prediction_try
from sklearn.linear_model import LinearRegression
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
    def form_network(self,n,m=2):
        network=[[{'weights':[0.42,-0.3998654,0.47]}]for i in range(n)]
        return network;
    #formation of our network
    def u_step(self,Z):
        thr=0.47
        if(Z>=thr):
            return 1.0;
        else:
            return 0.001;
    #Defining the unit step function
    def feed_forward(self,network,data):
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
                rown['cost_with_activation']=self.u_step(Z);
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
print(network);
print("========================================");
print("++++++++++++++++++++++++++++++++++++++++");
'''data=[[0.74,0.26,1],
      [0.13,0.87,0],
      [0.65,0.35,1],
      [0.98,0.02,1],
      [0.05,0.95,0],
      [0.5,0.5,1],
      [0.6,0.4,1],
      [0.2,0.8,0],
      [0.02,0.98,0],
      [0.38,0.62,0]]'''
data=list();l=0;
for i in range(1,11):
    x=exp(-i);
    if(x>0.47):
        l=1;
    else:
        l=0
    data.append([x,1-x,l])
n_data=[];
for i in data:
    n_data.append(i[0]);
t=prediction_try.cumulative(n_data);
n_data=np.array(t)
x=np.reshape(np.array([*range(1,11)]),(10,1))
network=s.feed_forward(network,data);
print(network);
s.plot(network,data)
regressor=LinearRegression()
regressor.fit(x,n_data)
l=regressor.predict([[11]]);
print(l)
t.append(l[0])
plt.figure(2)
prediction_try.plotting(t,12);
print("predicted happiness value=",(l[0]-n_data[-1]))
print("After passing through unit step activation:",s.u_step(l[0]-n_data[-1]));       
