
# coding: utf-8

# # Dog Breed Identification (Part I): Data Exploration
# 
# We continue with our efforts on the [Dog Breed Identification Kaggle Competition](https://www.kaggle.com/c/dog-breed-identification). This time, we'll focus on building a deep __convolutional neural network__ to model the function between input images, $\{ \mathbf{X}^{(i)} \}_{i = 1}^{N_{\textrm{train}}}$, and corresponding labels, $\{ y^{(i)} \}_{i = 1}^{N_{\textrm{train}}}$, where $N_{\textrm{train}}$ = the number of _training data_ available. The labels $y^{(i)} \in \{ 1, ..., K \}$ are proxy to the _object category_ of the input $\mathbf{X}^{(i)}$; e.g., if some $\mathbf{X}^{(i)}$ is an image containing an [Alaskan Malamute](https://en.wikipedia.org/wiki/Alaskan_Malamute), its label should be the integer which is mapped to the category "Alaskan Malamute". All possible object categories have a corresponding integer label.
# 
# We also have access to a (smaller) _validation dataset_, which we periodically evaluate our network on to ensure we don't [overfit](https://en.wikipedia.org/wiki/Overfitting) the training data. We denote the validation data as $(\{ \mathbf{X}^{(i)}, y^{(i)} \})_{i = 1}^{N_{\textrm{validate}}}$, where $N_{\textrm{validate}}$ = the number of _validation data_ available, and $N_{\textrm{validate}} << N_{\textrm{train}}$.
# 
# A neural network can be expressed as a parametric function $f(\mathbf{X}^{(i)}; \theta)$; parametrized by a _parameter vector_ $\theta$. The parameters correspond to the __learned weights__ on the connections between neurons of adjacent layers. The goal is to learn a setting of $\theta$ which minimizes the differences between $\sum_{i = 1}^{N_{\textrm{train}}} \left[ f(\mathbf{X}^{(i)}; \theta) - y^{(i)} \right]$, while simultaneously choosing a reasonable setting of parameters that we expect to generalize well to new (e.g., test) data.
# 
# For the Kaggle competition, we are given _test data_ $\{ \mathbf{X}^{(i)} \}_{i = 1}^{N_{\textrm{test}}}$, and we submit $\{ f(\mathbf{X}^{(i)}; \hat{\theta}^*) = \hat{y}^{(i)} \}_{i = 1}^{N_{\textrm{test}}}$, where $\hat{\theta}^*$ is an estimate of optimal parameters given the particular neural network model. These __predictions__ (or __inferences__) will be compared with the [ground truth](https://en.wikipedia.org/wiki/Ground_truth) categorical labels, and we will be ranked according to the number of test data our model misclassified.
# 
# At the time of writing, the best __error rate__ listed on the competition's leaderboard is 0.313% (accuracy of 100% - 0.313% = 99.687%). We don't expect to beat this, but obtaining a model with ~2-3% error rate is a realistic and challenging goal.

# ## Imports / miscellany

# In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

get_ipython().magic(u'matplotlib inline')

data_path = os.path.join('..', 'data')

# Are there CUDA-enabled GPU devices available? 
cuda = torch.cuda.device_count() > 0


# ## Load pre-processed doggos
# 
# We've already pre-processed the pupper image data. For now, we will simply load it into memory.

# In[2]:

# Load training (input, target) data.
X_train = np.load(os.path.join(data_path, 'X_train.npy')).transpose((0, 3, 2, 1))
y_train = np.load(os.path.join(data_path, 'y_train.npy'))
y_train = np.array([ np.argmax(y_train[idx, :]) for idx in range(y_train.shape[0]) ])

# Load validation (input, target) data.
X_valid = np.load(os.path.join(data_path, 'X_valid.npy')).transpose((0, 3, 2, 1))
y_valid = np.load(os.path.join(data_path, 'y_valid.npy'))
y_valid = np.array([ np.argmax(y_valid[idx, :]) for idx in range(y_valid.shape[0]) ])


# In[3]:

# Sanity check: print out training, validation data shapes
print('Training data shapes (X, y):', (X_train.shape, y_train.shape))
print('Validation data shapes (X, y):', (X_valid.shape, y_valid.shape))


# ## Define PyTorch neural network model
# 
# Here is where things get interesting. We will use the [PyTorch deep learning library](http://pytorch.org/) (which I highly recommend!) to create a convolutional neural network (CNN) to learn an approximate mapping between inputs $\mathbf{X}$ and targets $y$.

# In[9]:

class CNN(nn.Module):
    '''
    Defines the convolutional neural network model.
    '''
    def __init__(self, input_size, n_classes):
        '''
        Constructor for the CNN object.
        
        Arguments:
            - input_size (int): The number of units in the input "layer".
                Corresponds to the number of pixels in the input images.
            - n_classes (int): The number of target categories in the data.
        
        Returns:
            - Instantiated CNN object.
        '''
        super(CNN, self).__init__()
        
        # Convolutional layer portion of the network.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,                                 kernel_size=3, stride=1, padding=1)
        self.mpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,                                 kernel_size=3, stride=1, padding=1)
        self.mpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,                                 kernel_size=3, stride=1, padding=1)
        self.mpool3 = nn.MaxPool2d(kernel_size=2)
        
        # Fully-connected layer portion of the network.
        self.dense1 = nn.Linear(2048, 1024)
        self.dense2 = nn.Linear(1024, 256)
        self.dense3 = nn.Linear(256, n_classes)
    
    def forward(self, x):
        '''
        Defines the forward pass of the network.
        
        Arguments:
            - x (np.ndarray): A minibatch of images with
                shape (M, D, H, W), with M = minibatch size.
        
        Returns:
            - Activations of the final layer of the CNN. That is,
                the representation of the input, learned by the network,
                which is used to disentangle the correct object category.
        '''
        # First convolutional block (Conv2d -> MaxPool2d -> ReLU nonlinearity)
        x_ = F.relu(self.mpool1(self.conv1(x)))
        print(x_.size())
        
        # Second convolutional block (Conv2d -> MaxPool2d -> ReLU nonlinearity)
        x_ = F.relu(self.mpool2(self.conv2(x_)))
        print(x_.size())
        
        # Third convolutional block (Conv2d -> MaxPool2d -> ReLU nonlinearity)
        x_ = F.relu(self.mpool3(self.conv3(x_)))
        print(x_.size())
        
        # Flatten 3-dimensional output of last
        # convolutional block to be 1-dimensional.
        x_ = x_.view(-1, 1)  # Quirky!
        print(x_.size())
        
        # First fully-connected block (Linear -> ReLU)
        x_ = F.relu(self.dense1(x_))
        print(x_.size())
        
        # Second fully-connected block (Linear -> ReLU)
        x_ = F.relu(self.dense2(x_))
        print(x_.size())
        
        # Third fully-connected block (Linear -> ReLU)
        x_ = F.relu(self.dense3(x_))
        print(x_.size())


# Now that we've defined the network, we can instantiate one and train it to identify the dog breeds in our image dataset. But first, we must choose network [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) (parameters that are chosen prior to training which affect how the training operates, as opposed to parameters which are learned during network training). We also store some useful information in workspace-level variables, which can be changed once (here) to affect the rest of the notebook. Finally, we will use `torch.utils.data.DataLoader`s to simplify the presentation of data to the network.

# In[10]:

# Hyperparameters
n_epochs = 10  # No. of times to train on the entire training data.
batch_size = 100  # No. of examples used in minibatch stochastic gradient descent (SGD).
print_interval = 50  # No. of minibatch SGD iterations between each progress message.

# Useful information
input_size = (256, 256)  # As defined in "Data Exploration.ipynb".
n_classes = 120  # No. of doggo breeds.

# Cast training, validation data to `torch.Tensor`s.
try:
    X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train)
    X_valid, y_valid = torch.from_numpy(X_valid).float(), torch.from_numpy(y_valid)
except RuntimeError:
    print('Data already cast to torch.Tensor data type.')
    
# Data loaders
train_loader = DataLoader(dataset=TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=True)


# ## Train the doggo recognizer
# 
# We're ready to begin network training. We'll instantiate a network, define the criterion we aim to minimize ([Multiclass Log Loss](https://www.kaggle.com/wiki/MultiClassLogLoss)), define the optimization algorithm (we'll use [Adam](https://arxiv.org/abs/1412.6980), short for adaptive moments, a variant of SGD which dynamically updates individual parameter learning rates during training), and train the model for some number of epochs (the number of passes through the training data, given by `n_epochs`).

# In[11]:

# Instantiate CNN object.
network = CNN(input_size=input_size, n_classes=n_classes)
if cuda:
    network.cuda()

# Create loss / cost /objective function.
criterion = nn.CrossEntropyLoss()

# Specify optimization routine.
optimizer = torch.optim.Adam(network.parameters())


# Here is the training loop!

# In[ ]:

for epoch in range(n_epochs):
    # On each minibatch SGD iteration, we get `batch_size` samples from `X_train`.
    for idx, (inputs, targets) in enumerate(train_loader):
        # Convert `torch.Tensor`s to `Variable`s.
        if cuda:
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
        else:
            inputs = Variable(inputs)
            targets = Variable(targets)
        
        # Run forward, backward pass of network
        optimizer.zero_grad()  # zeros out gradient buffer
        predictions = network.forward(inputs)  # run forward pass of network to get predictions
        loss = criterion(predictions, targets)  # calculate loss (fn. of predictions and true targets)
        loss.backward()  # run backward pass (calculate grads. of loss w.r.t. network parameters)
        
        # Take optimization step (update network parameters in opposite direction of loss).
        optimizer.step()
        
        if i % print_interval == 0:
            print('Epoch [%d / %d], Iteration [%d / %d], Loss: %.4f' %                       (epoch + 1, n_epochs, idx + 1, ))

