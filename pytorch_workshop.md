  # Pytorch Workshop (UC Berkeley)
  
  https://github.com/mlberkeley/pytorch-workshop
  
  ### Slides
  
 https://github.com/mlberkeley/PyTorch-Workshop/blob/master/PyTorch_Workshop_slides.pdf
 
 1. Neural Network (Vector inputs, matrix multiplication, etc)
 - softmax(relu(w2 relu)
 
 2. We want a framework that makes sense 
 - General as possible, modular, transfer learning, general computation
 
 3. Define by Run
 - Tensorflow -> Computation Graph feed data through it, update in session
 - Pytorch -> dynamic computation graph -> dynamic inputs
 
 4. PyTorch Objects
 - Tensors for data and gradients 
 - autograd.Variable - data, grad, creator
 - creator -> pointer -> 
 - Tensor type -> gradients, still useful -> numpy arrays -> CUDA (Data and Grad in autograd.Variable)
 - grad for gradient descent 
 
 5. Computation Graph
 - how it is created on the fly
 - from torch.autograd import Variable
 - edges point where it was created, 
 
 6. First example -> skeleton
 https://github.com/mlberkeley/PyTorch-Workshop/blob/master/pytorch_mnist_skeleton.ipynb
 - Stochastic Gradient Descent
 - Loading and preprocessing data (a minute or two) MNIST -> constants, more understanding normalization
 - Write code in Pytorch write in classes in nn.module neural net module
 - nn.module -> inherited from
 - input channel (1 input, 10 channel, kernet_size)
 - Maxpool -> convolution -> information spatially into statistics information, invariant
 - Maxpool convolution -> be invariant where it is, so we're not fooled
 - Dropout probability (default = 0.5) 
 - Default padding -> convolutions stride and dialation (1 and 0) 
 - Look up square kernel, pass in a tuple
 - Dropout -> Regularize -> probability of dropping out to not move forward, forces others layers generalization 
 - Why do right self. (init defines all of these structure, forward pass in data model)
 - Pytorch - has autodifferentiation, backprop is done automatatically through forward
 - x.view (reshaping -> -1 infer that size, batch size 
 - where is getting 320, convolution layer (2-D -> 1-D reshape, size be generic), remove dimensionality, different padding, diff. kernel size,
 
  

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool1act = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool2act = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.activationfc = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.pool1act(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.pool2(x)
        x = self.pool2act(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.activationfc(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

7. Getting model ready to run
- 
model = Net()
if cuda:
  model.cuda()
optimizer = optim.SGD(model.parameters(), lr-lr, momentum=momentum)

8. Training

- model.train() -> dropou
- use cuda if possible
- find loss and backprop
- optimizer -> updates weight
- print stuff out, boilerplate
- quirks about PyTorch
- cuda methods (tensor and objects) 
- pytorch generalizations -> LSTMs batch size, weird permuate through batching

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

9. Testing the model

- optimzer, params weights
- PyTorch -> Don't shoot yourself in the foot
- Writing things quickly and building normal neural networks
- 99% MNIST, PyTorch, define by run

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 
for epoch in range(1, epochs + 1):
  train(epoch)
 
