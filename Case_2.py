import numpy as np
from math import exp,pow
from random import random,seed
from matplotlib import pyplot as plt
import prediction_try
from sklearn.linear_model import LinearRegression
from math import sqrt
import sys
import pandas as pd
from check_inv_cont import MappingEmotion
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
class Dynamics_Case3:
    def weights(self,w1,w2,w3):
        return np.array([w1,w2,w3]);
    def weighted_sum(self,row,w):
        return np.dot(w.T,row)
    #caluculating the weighted sum
    def sigmoid(self,Z):
        return 1/(1+exp(-Z))
    #function to find the sigmoid value
    def transfer_function(self,Z):
        s=self.sigmoid(Z);
        return s*(1-s);
    def form_network(self,n,m=0):
        if(m==1):
                network=[[{'weights':[0.5,0.5,0]}]for i in range(n)]
                return network; 
        network=[[{'weights':[0.5,-0.5,0.3]}]for i in range(n)]
        return network;
    def u_step(self,Z,val=0.3):
        thr=val
        if(Z>=thr):
            return 1.0;
        else:
            return 0.001;
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
    def continuity(self,mean,sd,Z):
        x=Z-mean;
        x=x/(2*pow(sd,2));
        expr=exp(-x)/(sd*pow(2*np.pi,0.5));
        return expr
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
s=Dynamics_Case3()
si=MappingEmotion()
data_A_B=[]
network=s.form_network(10)
#print(network);
for i in range(10):
        l=random()
        if(l>=0.3):
                f=1
        else:
                f=0
        data_A_B.append([l,1-l,f])
data_B_C=[];
for i in range(10):
        l=random()
        if(l>=0.3):
                f=1
        else:
                f=0
        data_B_C.append([l,1-l,f])
data_A_C=[];
for i in range(10):
        l=random()
        if(l>=0.3):
                f=1
        else:
                f=0
        data_A_C.append([l,1-l,f])
time=np.reshape(np.array([*range(1,11)]),(10,1))
for i in range(10):
        print(data_A_C[i][0])
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
n_data3=[]
for i in data_A_C:
    n_data3.append(i[0]);
ti2=prediction_try.cumulative(n_data3);
n_data3=np.array(ti)
#------------------------------------------------------------------------------------
network=s.feed_forward(network,data_A_B);
#s.plot(network,data_A_B)
regressor=LinearRegression()
regressor.fit(time,n_data)
l=regressor.predict([[11]])
t.append(l[0])
plt.figure(2)
prediction_try.plotting(t,12);
network=s.feed_forward(network,data_A_C);
#s.plot(network,data_A_C)
regressor.fit(time,n_data2)
l=regressor.predict([[11]])
ti.append(l[0])
plt.figure(2)
prediction_try.plotting(ti,12);
network=s.feed_forward(network,data_B_C);
#s.plot(network,data_B_C)
regressor.fit(time,n_data3)
l=regressor.predict([[11]])
ti2.append(l[0])
plt.figure(2)
prediction_try.plotting(ti2,12);
#-------------------------------------------------------------------------------------
print("CONSIDERING OUR PERSON 'C' is the Chatbot");
friend=check_greater_friend(data_A_C,data_B_C)
#-------------------------For A our response should be------------------------------------
print("A response")
w1=friend[0]
x1=ti[-2]
Z_new=w1*x1+ti[-1]
value_emotion_A=si.inv_continuous(Z_new,ti[-1],ti[-2])
print("The emotion values for A from C is as follows:");
print(value_emotion_A);
#---------------------------For B our response should be----------------------------------
print("A response")
w1=friend[0]
x1=ti2[-2]
Z_new=w1*x1+ti2[-1]
value_emotion_B=si.inv_continuous(Z_new,ti2[-1],ti2[-2])
print("The emotion values for B from C is as follows:");
print(value_emotion_B);
#---------------------------------------------------------------------------------------------
