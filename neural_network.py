import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    sig_val=1/(1+np.exp(-x))
    return sig_val
#input_value=np.random.rand(5,10,4)
#print(input_value.shape[1])
#print(input_value)
#input_value=np.linspace(-10,9.99,100)

#print(input_value)

#output_value=sigmoid(input_value)
#print(output_value)
#plt.plot(input_value,output_value)
#plt.xlabel("Input")
#plt.ylabel("Output")
#plt.title("Sigmoind function")
#plt.show()

input_features=np.array([[0,0],[0,1],[1,0],[1,1]])
target_output=np.array([0,1,1,1])
#output_features=sigmoid(input_features)
#print(target_output)
#print(target_output.shape)
#print(input_features)
#print(input_features.shape)
weights1=[]
weights1=np.array(np.random.rand(2))

#print("weights1:",weights1)

#weights=weights1.reshape(2,1)
#print(weights1.shape)

#print(weights1.T)

#print(weights1.T.shape)

bias=0.3

learning_rate=0.05

def d_sigmoid(x):
    in_value=np.dot(x,weights1)+bias
    return sigmoid(in_value)*(1-sigmoid(in_value))
doutput_value=d_sigmoid(input_features)
#print(d_sigmoid(input_features))

#def dcost(x):
   # dcost_output=-(output_features-target_output)

output_features=sigmoid(np.dot(input_features,weights1)+bias)
#plt.plot(input_features,output_features)
#plt.xlabel("Input")
#plt.ylabel("Output")
#plt.title("Sigmoind function ")
#plt.show()
#print(output_features)



#plt.plot(input_features,doutput_value)
#plt.xlabel("Input")
#plt.ylabel("DOutput")
#plt.title("Sigmoind function derivative")
#plt.show()
print("----------------------------------------------")

#for loop in range(1000):
#in_value=np.dot(input_features,weights1)+bias
#print(in_value)
def dcost(x):
    in_value=np.dot(x,weights1)+bias
    #print(in_value)
    dcost_output=(sigmoid(in_value)-target_output)
    return dcost_output
#print(in_value)
#print("Error cost output :",dcost)
#pltdcost.plot(input_features,in_value)
#plt.xlabel("Input")
#plt.ylabel("Output")
#plt.title("matrix input")
#plt.show()

print("Error cost output :")

print(dcost(input_features))

#print(dcost(input_features).sum())

print(output_features)
def backpropagation(x):
    weights=[]
    weights=weights1
    bias_temp=bias
    print("sigoid function:")
    print()
    d_sigmoid(x)
    dcost(x)
    print( "dcost:",dcost(x))
    cost_error=dcost(x)
    #cost_error=dcost(x).sum()
    print("cost error",cost_error)
    print(" derivative sigmod x:",d_sigmoid(x))
    deriv_z=cost_error*d_sigmoid(x)
    deriv_w=np.dot(x.T,deriv_z)
    weights-=learning_rate*deriv_w
    for i in deriv_z:
        bias_temp-=learning_rate*i
    #bias=bias_temp
    #weights1=weights
    return deriv_w,bias_temp,weights
div_w,bias,weights1=backpropagation(input_features)

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print(backpropagation(input_features))
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("bias is: ",bias)
print("the weights is : ",weights1)

output_features=sigmoid(np.dot(input_features,weights1)+bias)
print(output_features)
