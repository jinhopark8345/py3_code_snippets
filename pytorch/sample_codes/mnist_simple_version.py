import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


########################################################################
# 2. Define a Convolutional Neural Network
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.
import torch.optim as optim
criterion = nn.CrossEntropyLoss()

logsoftmax = nn.LogSoftmax()
nll_loss = nn.NLLLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network

breakpoint()
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        """
        logsoftmax = nn.LogSoftmax()
        nll_loss = nn.NLLLoss()

        tmp_loss = nll_loss(logsoftmax(outputs), labels)
        assert tmp_loss == loss

(Pdb++) logsoftmax(outputs)
tensor([[-2.2791, -2.1891, -2.2654, -2.3236, -2.3643, -2.3952, -2.2406, -2.4060,
         -2.3748, -2.2153],
        [-2.2821, -2.1823, -2.2668, -2.3189, -2.3718, -2.3939, -2.2452, -2.4083,
         -2.3731, -2.2120],
        [-2.2756, -2.1918, -2.2642, -2.3145, -2.3587, -2.4003, -2.2519, -2.4055,
         -2.3726, -2.2170],
        [-2.2811, -2.1948, -2.2630, -2.3196, -2.3671, -2.3961, -2.2438, -2.4039,
         -2.3727, -2.2108]], grad_fn=<LogSoftmaxBackward0>)

(Pdb++) torch.exp(outputs)
tensor([[1.0364, 1.1341, 1.0508, 0.9914, 0.9518, 0.9229, 1.0772, 0.9129, 0.9419,
         1.1047],
        [1.0322, 1.1406, 1.0481, 0.9950, 0.9437, 0.9231, 1.0711, 0.9099, 0.9424,
         1.1072],
        [1.0397, 1.1306, 1.0517, 1.0001, 0.9569, 0.9178, 1.0647, 0.9131, 0.9436,
         1.1026],
        [1.0329, 1.1260, 1.0517, 0.9939, 0.9478, 0.9207, 1.0721, 0.9136, 0.9425,
         1.1081]], grad_fn=<ExpBackward0>)

(Pdb++) torch.sum(torch.exp(outputs), dim=-1)
tensor([10.1240, 10.1133, 10.1207, 10.1093], grad_fn=<SumBackward1>)

(Pdb++) torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)
tensor([[10.1240],
        [10.1133],
        [10.1207],
        [10.1093]], grad_fn=<UnsqueezeBackward0>)

(Pdb++) torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)
tensor([[0.1024, 0.1120, 0.1038, 0.0979, 0.0940, 0.0912, 0.1064, 0.0902, 0.0930,
         0.1091],
        [0.1021, 0.1128, 0.1036, 0.0984, 0.0933, 0.0913, 0.1059, 0.0900, 0.0932,
         0.1095],
        [0.1027, 0.1117, 0.1039, 0.0988, 0.0945, 0.0907, 0.1052, 0.0902, 0.0932,
         0.1089],
        [0.1022, 0.1114, 0.1040, 0.0983, 0.0938, 0.0911, 0.1061, 0.0904, 0.0932,
         0.1096]], grad_fn=<DivBackward0>)

(Pdb++) torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1))
tensor([[-2.2791, -2.1891, -2.2654, -2.3236, -2.3643, -2.3952, -2.2406, -2.4060,
         -2.3748, -2.2153],
        [-2.2821, -2.1823, -2.2668, -2.3189, -2.3718, -2.3939, -2.2452, -2.4083,
         -2.3731, -2.2120],
        [-2.2756, -2.1918, -2.2642, -2.3145, -2.3587, -2.4003, -2.2519, -2.4055,
         -2.3726, -2.2170],
        [-2.2811, -2.1948, -2.2630, -2.3196, -2.3671, -2.3961, -2.2438, -2.4039,
         -2.3727, -2.2108]], grad_fn=<LogBackward0>)

(Pdb++) nll_loss(torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)), labels)
tensor(2.2976, grad_fn=<NllLossBackward0>)

(Pdb++) torch.gather(torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)), 1, labels.unsqueeze(dim=-1))
tensor([[-2.2406],
        [-2.1823],
        [-2.4003],
        [-2.3671]], grad_fn=<GatherBackward0>)

(Pdb++) torch.mean(torch.gather(torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)), 1, labels.unsqueeze(dim=-1)))
tensor(-2.2976, grad_fn=<MeanBackward0>)

(Pdb++) -torch.mean(torch.gather(torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)), 1, labels.unsqueeze(dim=-1)))
tensor(2.2976, grad_fn=<NegBackward0>)

(Pdb++) loss
tensor(2.2976, grad_fn=<NllLossBackward0>)




(Pdb++) criterion(outputs, labels)
tensor(2.2976, grad_fn=<NllLossBackward0>)

(Pdb++) nll_loss(logsoftmax(outputs), labels)
tensor(2.2976, grad_fn=<NllLossBackward0>)

(Pdb++) torch.exp(outputs).shape
torch.Size([4, 10])
(Pdb++) torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1).shape
torch.Size([4, 1])

(Pdb++) softmax_output = torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)
(Pdb++) nll_loss(torch.log(softmax_output), labels)
tensor(2.2976, grad_fn=<NllLossBackward0>)


        """
        breakpoint()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

########################################################################
# Let's quickly save our trained model:

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

########################################################################
# 5. Test the network on the test data
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = Net()
net.load_state_dict(torch.load(PATH))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

