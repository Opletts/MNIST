import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_data = datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root="./data/", train=False, transform=transforms.ToTensor(), download=True)

train_load = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_load = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

def display(loader, n, train = True):
	dataiter = iter(loader)
	img, label = dataiter.next()
	if train == False:
		outputs = net(Variable(img))

	for i in range(n):
		test = img[i].numpy()
		if train == False:
			pred = torch.topk(outputs[i], 1)[1].data[0]
			print "Label : {0} Prediction : {1}".format(label[i], pred)
		else:
			print "Label : {}".format(label[i])
		cv2.imshow("Digit", np.squeeze(test))
		cv2.waitKey(0)

	cv2.destroyAllWindows()

# display(train_load, 5)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 1, padding = 2) # W_in = 28 W_out = 28  (W_in - F + 2 * P) / S 	+	1
		self.mx = nn.MaxPool2d(kernel_size = 2, stride = 2) 					# W_in = 28 W_out = 14
		self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
		self.dout = nn.Dropout()
		self.fc1 = nn.Linear(7 * 7 * 64, 1000)								#max_pool again after conv2, becomes W_out = 7
		self.fc2 = nn.Linear(1000, 10)		

	def forward(self, x):
		x = F.relu(self.mx(self.conv1(x)))
		x = F.relu(self.mx(self.conv2(x)))
		x = x.view(-1, 7 * 7 * 64)
		x = self.dout(x)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.5)

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print "Total parameters : {}".format(pytorch_total_params)

def train(epochs):
	net.train()
	for epoch in range(epochs):
		for i, data in enumerate(train_load):
			inputs, labels = data

			outputs = net(Variable(inputs))
			loss = criterion(outputs, Variable(labels))

			if i%100 == 0:
				print "Epoch : {} Step : {} Loss : {}".format(epoch, i, loss.data[0])

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print "Train Accuracy : {} Test Accuracy : {}".format(eval(train_load), eval(test_load))
		torch.save(net, "mnist_cnn")

	print "Training Done"

def eval(load):
	correct = 0
	total = 0
	net.eval()
	for data in load:
		inputs, labels = data

		outputs = net(Variable(inputs))

		_, pred = torch.max(outputs, 1)

		total += labels.size(0)
		correct += torch.sum(pred == Variable(labels)).data[0]

	return float(correct) * 100 / total

train(10)
eval(test_load)

# torch.save(net, "mnist_nn")
display(test_load, 10, 0)