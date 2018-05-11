import numpy as np 
'''

Input _layer(32 X 32) ==> convolution_layer(kernel = [3 X 3 X 4],stride = [2 X 2])(16 X 16 X 4) ==> relu ==> Fully_connected(1024 X 10) ==> sigmoid ==> output(10 X 1) 

'''


class Variables:#initialization
	def __init__(self,variable):# Initialization of Variable 
		self.variable = variable
	def update(self,change):# Update variable during learning  
		self.variable = change
		return 0

def sigmoid(activation_l):# activation function for the last layer 
	temp = []	
	for i in activation_l:
		temp.append(1/(1 + np.exp(-i)))
	return np.array(temp)	

def Get_Gradient_sigmoid(activation_l):
	temp = []
	for i in np.array(activation_l):# Derivative of sigmoid = sigmoid*(1 - sigmoid)
		temp.append((1/(1 + np.exp(- i)))*(1 - 1/(1 + np.exp(- i))))	
	return 	np.array(temp)

def Get_Gradient_relu(activation_l):
	'''
	Derivative of relu = {
				1 : x > 0
				0 : x <= 0

				}		
	'''

	return (activation_l>0)*1

	
class error:

	def __init__(self,labels,logits):# Cost function = 1/2*sum((labels - logits)^2)
		a = []
		
		for i in range(labels.shape[0]):	
			a.append(np.sum((labels[i] - logits[i])**2)/2)
			
		self.loss = a
		

	def Find_error(self,labels,logits):# Derivative of cost function = sum((labels - logits))
		arr = []

		for i in range(labels.shape[0]):	
			
			arr.append(np.sum(labels[i] - logits[i]))
	
		self.Err = arr

		return 0
	def backward_FC(self,W,LR,Gradient,last_l):#Backpropagation for Fully connected layer
		'''
		W  : weights Tensor 
		LR : learning rate 
		last_l : last layer
		Gradient : derivative values of output layer
	
		Updating equations :
			Error = 1/2*(sum(Err^2)) , Err = (labels[i] - logits[i])
			dError/dW[i,j] = Err * dErr/dW[i,j]
			dError/dW[i,j] = Err[j] * (- Gradient[j]) *last_l[i] 
		Gradient decent :
 			W[i,j] <== W[i,j] + LR * dError/dW[i,j]
		'''
 
		for i in range(W.variable.shape[0]):
			for j in range(W.variable.shape[1]):
				W.variable[i,j] = W.variable[i,j] + LR*self.Err[j]*Gradient[j]*last_l[i]
				print(W.variable[i,j] + LR*self.Err[j]*Gradient[j]*last_l[i])
		return 0

	def backward_conv(self,W1,W2,LR,Gradient,Gradient2,last_l,H,W,n_output):
		'''
	
		W1 : weights Tensor of convolution layer
		W2 : weights Tensor of Fully connected layer 
		LR : learning rate 
		Gradient : derivative of output layer 
		Gradient2 : derivative of Hidden layer
		last_l : input layer
		H : number of rows of feature vector after convolution(16)
		W : number of columns of feature vector after convolution(16)
		n_output : number of rows of output _layer(10)
		
		updating equations :
			dError/dW1[i,j,k] = sum_p(Err[p] * dlogits[p]/dW1[i,j,k]) , 0 < p < 11
			dlogits[p]/dW1[i,j,k] = Gradient2[p] * (sum_i'(sum_j'(W2[i'*W + j, p] * last_l[i' + i,j' + j])))

		Gradient decent :
 			W[i,j,k] <== W[i,j,k] + LR * dError/dW[i,j,k]
		

		'''
		for k in range(W1.variable.shape[2]):
			for m in range(W1.variable.shape[0]):
				for n in range(W1.variable.shape[1]):		
					delta1 = 0
					for o in range(n_output):
						delta = 0
						for i in range(H):
							for j in range(W):
								delta += W2.variable[i*W + j,o]*last_l[i+m,j+n]*Gradient2[i*W + j]
						delta1 += self.Err[o]*Gradient[o]*delta
					W1.variable[m, n] = W1.variable[m, n] + LR*delta1
		return 0
	
def forward_conv(X,W,strides):#X is 2D, W = [row,columns,o_channels] , strides = [row,column]
	'''
		(I # K)[i,j] = sum_m(sum_n(sum_c(K[m,n,c] * I[strides[0] * i + m,strides[1] * j + n]))); 0 <= m <=k1-1,0 <= n <=k2-1,0 <= c <= D,
		K ϵ R^(k1 * k2 * C * D)
		I ϵ R^(H * W * C)
		# : convolutional operation
	'''

	activation = []
	for i in range(X.shape[0]//strides[0]):	
		temp2 = []
		for j in range(X.shape[1]//strides[1]): 
			temp1 = []
			for k in range(W.shape[2]):
				temp = X[2*i:2*i+W.shape[0] ,2*j:2*j+W.shape[1]]*W[:,:,k]
				temp = np.sum(np.sum(temp,axis=1),axis=0)
				temp1.append(temp)
			temp2.append(temp1)
			
		activation.append(temp2)	
	activation = activation*((activation>0)*1)					
	return np.array(activation)

def forward_FC(X,W):
	return np.matmul(X, W)

epoch = 100
# Variable declaration

W1 = Variables(np.ones([3,3,4]))
W2 = Variables(np.ones([1024,10]))

X = np.ones([33, 33])#input
Y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])#label
for n_epoch in range(epoch):
	h_conv1 = forward_conv(X,W1.variable,[2, 2])
	linear_vector = []

	for k in range(h_conv1.shape[2]):
		for i in range(h_conv1.shape[0]):
			for j in range(h_conv1.shape[1]):
				linear_vector.append(h_conv1[i,j,k])

	linear_vector = np.array(linear_vector)

	Y_pred = sigmoid(forward_FC(linear_vector,W2.variable))#output(logits)

	energy = error(Y, Y_pred)
	delta1 = Get_Gradient_sigmoid(Y_pred)
	delta2 = Get_Gradient_relu(linear_vector)
	energy.Find_error(Y, Y_pred)
	energy.backward_FC(W2,0.01,delta1,linear_vector)
	energy.backward_conv(W1,W2,0.01,delta1,delta2,X,h_conv1.shape[0],h_conv1.shape[1],10)
	print(Y_pred)



					
				

				
			
		
