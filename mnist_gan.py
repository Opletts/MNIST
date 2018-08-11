import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

train_data = datasets.MNIST(root="./data/", train=True, transform=compose, download=True)
test_data = datasets.MNIST(root="./data/", train=False, transform=compose, download=True)

train_load = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_load = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

def display(loader):
	stack = (loader[0].view(1, 28, 28)).data.numpy()
	stack = np.squeeze(stack)
	for i in range(8):
		img = (loader[i+1].view(1, 28, 28)).data.numpy()
		img = np.squeeze(img)
		stack = np.hstack((img, stack))
		cv2.imshow("Test", stack)
		cv2.waitKey(1)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.layer1 = nn.Sequential(
						nn.Linear(784, 1000),
						nn.LeakyReLU(0.2),
						nn.Dropout()
						)
		self.layer2 = nn.Sequential(
						nn.Linear(1000, 500),
						nn.LeakyReLU(0.2),
						nn.Dropout()
						)
		self.layer3 = nn.Sequential(
						nn.Linear(500, 200),
						nn.LeakyReLU(0.2),
						nn.Dropout()
						)
		self.layer4 = nn.Sequential(
						nn.Linear(200, 1),
						nn.Sigmoid()
						)

	def forward(self, x):
		x = x.view(-1, 784)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.layer1 = nn.Sequential(
						nn.Linear(100, 200),
						nn.LeakyReLU(0.2),
						nn.Dropout()
						)
		self.layer2 = nn.Sequential(
						nn.Linear(200, 500),
						nn.LeakyReLU(0.2),
						nn.Dropout()
						)
		self.layer3 = nn.Sequential(
						nn.Linear(500, 1000),
						nn.LeakyReLU(0.2),
						nn.Dropout()
						)
		self.layer4 = nn.Sequential(
						nn.Linear(1000, 784),
						nn.Tanh()
						)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x

disc = Discriminator()
gen = Generator()

criterion = nn.BCELoss()
optimizer_gen = optim.Adam(gen.parameters(), lr = 0.001)
optimizer_disc = optim.SGD(disc.parameters(), lr = 0.001, momentum = 0.5)

total_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
print "Discriminator parameters : {}".format(total_params)

total_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
print "Generator parameters : {}".format(total_params)

def train_gan(epochs):
	disc.train()
	gen.train()
	for epoch in range(epochs):
		for i, data in enumerate(train_load):
			
			inputs, _ = data
			labels = Variable(torch.ones(inputs.size(0)))

			outputs = disc(Variable(inputs))

			real_loss = criterion(outputs, labels)

			noise = Variable(torch.randn(inputs.size(0), 100))
			fake_inputs = gen(noise)

			fake_labels = Variable(torch.zeros(inputs.size(0)))

			outputs = disc(fake_inputs)

			fake_loss = criterion(outputs, fake_labels)

			loss = real_loss + fake_loss

			gen_loss = criterion(outputs, labels)

			if i%100 == 0:
				print "Discriminator Loss : {} Generator Loss : {}".format(loss.data[0], gen_loss.data[0])
				display(fake_inputs)

			optimizer_disc.zero_grad()
			loss.backward(retain_variables=True)
			optimizer_disc.step()

			optimizer_gen.zero_grad()
			gen_loss.backward()			
			optimizer_gen.step()

train_gan(1)