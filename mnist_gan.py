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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

disc = Discriminator()
gen = Generator()

criterion = nn.BCELoss()
optimizer_gen = optim.Adam(gen.parameters(), lr = 0.001)
optimizer_disc = optim.SGD(disc.parameters(), lr = 0.001, momentum = 0.5)

total_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
print "Discriminator parameters : {}".format(total_params)

total_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
print "Generator parameters : {}".format(total_params)

def train_disc(real_imgs, fake_imgs, real_labels, fake_labels):
	disc.train()
	optimizer_disc.zero_grad()
	real_outputs = disc(real_imgs)
	real_loss = criterion(real_outputs, real_labels)

	fake_outputs = disc(fake_imgs)
	fake_loss = criterion(fake_outputs, fake_labels)

	loss = real_loss + fake_loss

	loss.backward()
	optimizer_disc.step()

	return loss

def train_gen(fake_inputs):
	gen.train()
	optimizer_gen.zero_grad()

	outputs = disc(fake_inputs)

	real_labels = Variable(torch.ones(real_imgs.size(0)))
	real_labels = real_labels.to(device)

	gen_loss = criterion(outputs, real_labels)

	gen_loss.backward()
	optimizer_gen.step()

	return gen_loss


epochs = 100

for epoch in range(epochs):
	for i, data in enumerate(train_load):
		real_imgs, _ = data
		real_imgs = real_imgs.to(device)

		disc_noise = Variable(torch.randn(real_imgs.size(0), 100))
		disc_noise = disc_noise.to(device)

		fake_imgs = gen(disc_noise).detach()

		real_labels = Variable(torch.ones(real_imgs.size(0)))
		real_labels = real_labels.to(device)

		fake_labels = Variable(torch.zeros(real_imgs.size(0)))
		fake_labels = fake_labels.to(device)

		disc_loss = train_disc(Variable(real_imgs), fake_imgs, real_labels, fake_labels)

		gen_noise = Variable(torch.randn(real_imgs.size(0), 100))
		gen_noise = gen_noise.to(device)

		gen_img = gen(gen_noise)
		gen_img = gen_img.to(device)

		g_loss = train_gen(gen_img)

		if i%100==0:
			print "Discriminator Loss : {} Generator Loss : {}".format(disc_loss.data[0], g_loss.data[0])