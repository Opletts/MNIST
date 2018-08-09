# MNIST
Digit classification using PyTorch.

Run `vanilla_nn.py` to get a trained model.
Load the model in `draw.py`, draw your own digit and classify it.

`mnist_cnn` is a basic convnet. To load this model into `draw.py` you'll have to change the class structure and change 
`line 68` to
```img = np.reshape(img, (1, 1, 28, 28))```