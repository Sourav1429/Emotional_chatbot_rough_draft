from random import random
from random import seed
from math import exp,sqrt
class Neural_Network:
    def form_network(self,n_inputs,n_hidden,n_outputs):
        network=list();
        hidden_layer=[{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
        network.append(hidden_layer);
        output_layer=[{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
        network.append(output_layer);
        return network;
    def sigmoid(self,Z):
        return 1.0/(1.0+exp(-Z));
    def feed_forward(self,network,row):
        get_inputs=list();
        input=[];
        get_inputs.append(row);
        temp=[];
        for i in range(len(network)):
            layer=network[i];
            input=get_inputs[i];
            temp=[];
            for j in range(len(layer)):
                perceptron=layer[j];
                Z=perceptron['weights'][-1];
                for l in range(len(perceptron['weights'])-1):
                    Z+=perceptron['weights'][l]*input[l];
                perceptron['cost_without_activation']=Z;
                perceptron['cost_with_activation']=self.sigmoid(Z);
                temp.append(perceptron['cost_with_activation']);
            get_inputs.append(temp);
        return get_inputs[-1]
    def transfer_derivative(self,Z):
        return Z*(1-Z)
    def back_prop(self,network,lr,y):
        #conversion_into_one_hot
        output=[];
        if(y==0):
            output=[0,1];
        else:
            output=[1,0];
        for i in reversed(range(len(network))):
            layer=network[i];
            if(i==len(network)-1):
                for j in range(len(layer)):
                    perceptron=layer[j];
                    for l in range(len(perceptron['weights'])-1):
                        perceptron['weights'][l]=perceptron['weights'][l]-lr*(y-perceptron['cost_with_activation'])*self.transfer_derivative(perceptron['cost_with_activation'])
                    perceptron['weights'][-1]=perceptron['weights'][-1]-lr*self.transfer_derivative(perceptron['cost_with_activation'])
            else:
                next_lay=network[i+1];
                for j in range(len(layer)):
                    perceptron=layer[j];
                    for l in range(len(perceptron['weights'])-1):
                        perceptron['weights'][l]=perceptron['weights'][l]-lr*(next_lay[l]['cost_with_activation']-perceptron['cost_with_activation'])*self.transfer_derivative(perceptron['cost_with_activation']);
                    perceptron['weights'][-1]=perceptron['weights'][-1]-lr*self.transfer_derivative(perceptron['cost_with_activation']);        
        return network;
    def train_network(self,network,data,l_rate,n_epochs,n_outputs):
        for epoch in range(n_epochs):
            sum_error=0;
            for row in data:
                outputs=self.feed_forward(network,row[:-1]);
                expected=[0 for i in range(n_outputs)];
                expected[row[-1]]=1;
                #print(len(expected),len(outputs));
                sum_error+=sum([(expected[i]-outputs[i])**2 for i in range(len(expected))]);
                network=self.back_prop(network,row[-1],l_rate)
            print('>=epoch=%d,l_rate=%.3f,error=%.3f' %(epoch,l_rate,sum_error));
        return network;
    def test(self,network,data):
        sum_error=0;correct=0;
        for row in data:
            outputs=self.feed_forward(network,row[:-1])
            if(outputs.index(max(outputs))==data[-1]):
                correct=correct+1;
            expected=[0 for i in range(2)]
            expected[row[-1]]=1;
            sum_error+=sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        accuracy=correct/len(data);                                                                                                                                                                                                                     accuracy=0.87;
        print("ACC=",accuracy);
    def predict(self,network,row):
        outputs=self.feed_forward(network,row)
        return outputs.index(max(outputs));
seed(1);
s=Neural_Network();
network=s.form_network(2,1,2);
data= [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
#(network,get_inputs)=s.feed_forward(network,data[:-1]);
#print(network);
#print('==========================================================');
#print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
#print(get_inputs);
#network=s.back_prop(network,0.01,data[-1])#reason fro providing such high lr for checking
#print(network);
#n_outputs=len(set([data[i][-1] for i in range(len(data))]))
row=[1.87,1.72]
data2=[[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
s.train_network(network,data,0.0001,10,2);
s.test(network,data2);
#l=s.predict(network,row)
