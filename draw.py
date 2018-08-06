import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

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

ap = argparse.ArgumentParser()
ap.add_argument("--model", required = True, help = "Path to Model to be loaded")
args = ap.parse_args()

net = torch.load(args.model)
draw = False
thickness = 5

def nothing(x):
	pass

def draw_digit(event, x, y, flags, param):
	global draw, thickness
	if event == cv2.EVENT_LBUTTONDOWN:
		draw = True
		cv2.circle(img, (x,y), thickness, (255), -1)

	elif event == cv2.EVENT_MOUSEMOVE:
		if draw:

			cv2.circle(img, (x,y), thickness, (255), -1)

	elif event == cv2.EVENT_LBUTTONUP:
		if draw:
			draw = False


img = np.zeros((100, 100), np.uint8)

cv2.namedWindow('image')
cv2.createTrackbar('Thickness', 'image', 5, 30, nothing)
cv2.setMouseCallback('image',draw_digit)

while(1):
	cv2.imshow('image',img)
	thickness = cv2.getTrackbarPos('Thickness', 'image')
	k = cv2.waitKey(10)
	if k == ord('p'):		
		img = cv2.resize(img, (28,28))
		img = np.reshape(img, (-1, 28, 28))
		output = net(Variable(torch.from_numpy(img).type(torch.FloatTensor)))
		_, pred_tensor = torch.max(output, 1)
		pred = pred_tensor.data.numpy()[0][0]

		print "Prediction : {}".format(pred)
		img = np.zeros((100, 100), np.uint8)

	elif k == ord('c'):
		img = np.zeros((100, 100), np.uint8)

	elif k == ord('q'):
		break

cv2.destroyAllWindows()