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

def display(loader):
	dataiter = iter(loader)
	img, label = dataiter.next()
	# print label[0]
	# test = img[0].numpy()
	# cv2.imshow("test", np.squeeze(test))
	# cv2.waitKey(0)

	outputs = net(Variable(img))
	_, pred = torch.max(outputs, 1)
	print "Prediction : "
	print label.numpy()
	print pred.data.numpy().reshape(-1)

# for i in range(5):
# 	display()

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(784, 500)
		self.fc2 = nn.Linear(500, 250)
		self.fc3 = nn.Linear(250, 120)
		self.fc4 = nn.Linear(120, 60)
		self.fc5 = nn.Linear(60, 10)

	def forward(self, x):
		x = x.view(-1, 784)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = self.fc5(x)

		return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.5)

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print pytorch_total_params

steps = 3
for step in range(steps):
	for i, data in enumerate(train_load):
		inputs, labels = data

		outputs = net(Variable(inputs))
		loss = criterion(outputs, Variable(labels))

		if i%100 == 0:
			print " Loss : " + str(loss.data[0])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


print "Training Done"

# dataiter = iter(train_load)
# img, label = dataiter.next()
# output = net(Variable(img))
# _, pred = torch.max(output, 1)

# for i in range(5):
# 	disp = img[i].numpy()
# 	print "Label : " + str(label[i]) + ", Prediction : " + str(pred.data[i][0])
# 	cv2.imshow("Image", np.squeeze(disp))
# 	cv2.waitKey(1)

correct = 0
total = 0

for data in test_load:
	inputs, labels = data

	outputs = net(Variable(inputs))

	_, pred = torch.max(outputs, 1)

	total += labels.size(0)
	correct += torch.sum(pred == Variable(labels)).data[0]
print total, correct
print "Accuracy : " + str(float(correct) * 100 / total)