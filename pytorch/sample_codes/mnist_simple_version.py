import torch
import torchvision
import torchvision.transforms as transforms


torch.manual_seed(2023)

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
        x = self.pool(F.relu(self.conv1(x))) # [bsz, 3, 32, 32] -> [bsz, 6, 28, 28] -> [bsz, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x))) # [bsz, 6, 14, 14] -> [bsz, 16, 10, 10] -> [bsz, 16, 5, 5]
        x = torch.flatten(x, 1) # flatten all dimensions except batch # [bsz, 16, 5, 5] -> [bsz, 16 * 5 * 5], 16*25=400
        x = F.relu(self.fc1(x)) # [bsz, 400] -> [bsz, 120]
        x = F.relu(self.fc2(x)) # [bsz, 120] -> [bsz, 84]
        x = self.fc3(x) # [bsz, 84] -> [bsz, 10]
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

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        breakpoint()
        """
        logsoftmax = nn.LogSoftmax()
        nll_loss = nn.NLLLoss()

        tmp_loss = nll_loss(logsoftmax(outputs), labels)
        assert tmp_loss == loss

(Pdb++) outputs
tensor([[ 0.0615,  0.0059, -0.1052, -0.0125,  0.0449, -0.0324,  0.0117,  0.0059,
         -0.0604, -0.0742],
        [ 0.0445, -0.0172, -0.0899, -0.0381,  0.0661, -0.0428,  0.0254, -0.0319,
         -0.0471, -0.0638],
        [ 0.0454, -0.0169, -0.1020, -0.0253,  0.0635, -0.0609,  0.0238, -0.0156,
         -0.0533, -0.0596],
        [ 0.0454, -0.0192, -0.0937, -0.0316,  0.0656, -0.0517,  0.0291, -0.0304,
         -0.0510, -0.0676]], grad_fn=<AddmmBackward0>)

(Pdb++) outputs.shape
torch.Size([4, 10])

(Pdb++) logsoftmax(outputs)
tensor([[-2.2269, -2.2825, -2.3935, -2.3009, -2.2434, -2.3207, -2.2766, -2.2824,
         -2.3488, -2.3626],
        [-2.2397, -2.3014, -2.3741, -2.3223, -2.2181, -2.3270, -2.2588, -2.3161,
         -2.3314, -2.3480],
        [-2.2383, -2.3006, -2.3857, -2.3090, -2.2202, -2.3446, -2.2599, -2.2994,
         -2.3370, -2.3433],
        [-2.2379, -2.3025, -2.3770, -2.3149, -2.2177, -2.3349, -2.2542, -2.3137,
         -2.3343, -2.3509]], grad_fn=<LogSoftmaxBackward0>)

(Pdb++) torch.exp(outputs)
tensor([[1.0634, 1.0059, 0.9002, 0.9875, 1.0459, 0.9681, 1.0118, 1.0060, 0.9414,
         0.9285],
        [1.0455, 0.9830, 0.9140, 0.9626, 1.0683, 0.9581, 1.0258, 0.9686, 0.9539,
         0.9382],
        [1.0464, 0.9833, 0.9030, 0.9751, 1.0656, 0.9409, 1.0241, 0.9845, 0.9481,
         0.9421],
        [1.0464, 0.9809, 0.9105, 0.9689, 1.0678, 0.9496, 1.0295, 0.9701, 0.9503,
         0.9346]], grad_fn=<ExpBackward0>)

(Pdb++) torch.sum(torch.exp(outputs), dim=-1)
tensor([9.8587, 9.8180, 9.8130, 9.8088], grad_fn=<SumBackward1>)

(Pdb++) torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)
tensor([[9.8587],
        [9.8180],
        [9.8130],
        [9.8088]], grad_fn=<UnsqueezeBackward0>)

(Pdb++) torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1).shape
torch.Size([4, 1])

(Pdb++) output_softmax = torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)
(Pdb++) output_softmax
tensor([[0.1079, 0.1020, 0.0913, 0.1002, 0.1061, 0.0982, 0.1026, 0.1020, 0.0955,
         0.0942],
        [0.1065, 0.1001, 0.0931, 0.0980, 0.1088, 0.0976, 0.1045, 0.0987, 0.0972,
         0.0956],
        [0.1066, 0.1002, 0.0920, 0.0994, 0.1086, 0.0959, 0.1044, 0.1003, 0.0966,
         0.0960],
        [0.1067, 0.1000, 0.0928, 0.0988, 0.1089, 0.0968, 0.1050, 0.0989, 0.0969,
         0.0953]], grad_fn=<DivBackward0>)

(Pdb++) output_logsoftmax = torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1))
(Pdb++) output_logsoftmax
tensor([[-2.2269, -2.2825, -2.3935, -2.3009, -2.2434, -2.3207, -2.2766, -2.2824,
         -2.3488, -2.3626],
        [-2.2397, -2.3014, -2.3741, -2.3223, -2.2181, -2.3270, -2.2588, -2.3161,
         -2.3314, -2.3480],
        [-2.2383, -2.3006, -2.3857, -2.3090, -2.2202, -2.3446, -2.2599, -2.2994,
         -2.3370, -2.3433],
        [-2.2379, -2.3025, -2.3770, -2.3149, -2.2177, -2.3349, -2.2542, -2.3137,
         -2.3343, -2.3509]], grad_fn=<LogBackward0>)

(Pdb++) nll_loss(torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)), labels)
tensor(2.3258, grad_fn=<NllLossBackward0>)

(Pdb++) output_batch_loss = torch.gather(torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)), 1, labels.unsqueeze(dim=-1))
(Pdb++) output_batch_loss
tensor([[-2.3207],
        [-2.3480],
        [-2.2994],
        [-2.3349]], grad_fn=<GatherBackward0>)

(Pdb++) torch.mean(torch.gather(torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)), 1, labels.unsqueeze(dim=-1)))
tensor(-2.3258, grad_fn=<MeanBackward0>)

(Pdb++) -torch.mean(torch.gather(torch.log(torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=-1).unsqueeze(dim=-1)), 1, labels.unsqueeze(dim=-1)))
tensor(2.3258, grad_fn=<NegBackward0>)

(Pdb++) loss
tensor(2.3258, grad_fn=<NllLossBackward0>)

(Pdb++) criterion(outputs, labels)
tensor(2.3258, grad_fn=<NllLossBackward0>)

(Pdb++) nll_loss(logsoftmax(outputs), labels)
tensor(2.3258, grad_fn=<NllLossBackward0>)



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

