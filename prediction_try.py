import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
def cumulative(data):
    cumulative_sum=list();
    for i in range(len(data)):
        if(i==0):
            cumulative_sum.append(data[i]);
        else:
            cumulative_sum.append(cumulative_sum[i-1]+data[i]);
    return cumulative_sum;
def plotting(data,time=11):
    plt.plot(0,0)
    plt.plot([*range(1,time)],data)
    plt.scatter([*range(1,time)],data)
    if(time==12):
        plt.scatter(11,data[-1],s=100)
    plt.xlabel('Time Axis')
    plt.ylabel('Cumulative Sum')
    plt.show()
def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
##data=[]        
n_data=[
0.217902,
0.301669,
0.359001,
0.443554,
0.438552,
0.438579,
0.443785,
0.460764,
0.515794,
0.750023]
#data=[1,3,4,8,6,3,7,2,9,3]
s=cumulative(n_data);
##print(s);
###plotting(s);
###(m,c)=estimate_coef([*range(1,11)],data)
###print(m,c)
data=np.array(s)
data=np.reshape(data,(10,1));
##
x=[1,2,3,4,5,6,7,8,9,10]
x=np.reshape(np.array(x),(10,1))
regressor=LinearRegression()
regressor.fit(x,data)
l=regressor.predict([[11],[12]])
num=data[-1]
##print(num[0]);
##print(l)
##print(l[0][0]);
for i in range(len(l)):
    print(l[i][0]-num[0]);
    num=l[i]
    
##s.append(l[0])
##plotting(s,12);


