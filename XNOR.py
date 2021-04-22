from random import random
from math import exp
class CaseA:
    def sigmoid(self,Z):
        if(Z>=0 and Z<=1):
            return 1;
        else:
            return 0;
    def network(self,data):
        p1=[-1.7836139,1.7462262,0.21112037]
        p2=p1;
        p3=[1.2613548,0.9862008,-0.26548553]
        thr=0.45
        if(data[0]>=thr):
            data[0]=1;
        else:
            data[0]=0;
        if(data[1]>=thr):
            data[1]=1;
        else:
            data[1]=0;
        print(data)
        Z1=p1[0]*data[0]+p1[1]*data[1]+p1[2];
        #print(Z1)
        Z1=self.sigmoid(Z1);
        #print(Z1)
        Z2=p1[0]*(1-data[0])+p1[1]*(1-data[1])+p1[2]
        #print(Z2)
        Z2=self.sigmoid(Z2)
        #print(Z2)
        l=p3[0]*Z1+p3[1]*Z2+p3[2]
        if(l>=0):
            z=1;
            return [z,l,"Friendly"];
        else:
            z=0
            return [z,l,"Hostile"]
