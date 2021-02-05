import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
import matplotlib.pyplot as plt

###################
## PREPROCESSING ##
###################

num_workers = 0
batch_size = 128
dataset_path = '/data/jedrzej/medical/covid_dataset/'
# define transforms:
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# define datasets:
train_data = datasets.ImageFolder(dataset_path, transform=train_transform)

# TODO: delete these lines because they are unn
val_data = datasets.ImageFolder(dataset_path, transform=test_transform)
test_data = datasets.ImageFolder(dataset_path, transform=test_transform)

print(len(train_data))

num_train_ex = len(train_data)
indices = list(range(num_train_ex))
split = int(np.floor(0.3 * num_train_ex))
np.random.shuffle(indices)
train_idx, test_and_val_idx = indices[split:], indices[:split]
half_index = int(len(test_and_val_idx)/2)
test_idx, val_idx = test_and_val_idx[half_index:], test_and_val_idx[:half_index]
print('Initialized indices to shuffle')

train_data_1 = torch.utils.data.Subset(train_data, train_idx)
test_data = torch.utils.data.Subset(train_data, test_idx)
val_data = torch.utils.data.Subset(train_data, val_idx)

train_loader = torch.utils.data.DataLoader(train_data_1, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
print('Initialized loaders')

##############
## TRAINING ##
##############

# initialize the model:

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 3)
print(model)

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# specify scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# number of epochs to train the model
n_epochs = 30

# lists to keep track of training progress:
train_loss_progress = []
val_accuracy_progress = []

model.train() # prep model for training

n_iterations = int(len(train_data)/batch_size)

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################

    for iter, (data, target) in enumerate(train_loader):
        print("Epoch:", epoch, "Iteration:", iter, "out of:", n_iterations)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(data)
        # calculate the loss
        loss = criterion(outputs, target)
        
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    # if you have a learning rate scheduler - perform a its step in here
    scheduler.step()
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    train_loss_progress.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

    # Run the test pass:
    correct = 0
    total = 0
    model.eval()  # prep model for validation

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    val_acc = (100 * correct / total)
    print('Accuracy of the network on the validation set: %d %%' % (val_acc))
    val_accuracy_progress.append(val_acc)

#######################
## SAVING THE MODEL ##
#######################

# appending the date and time to automate renaming of file
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")

PATH = '/home/dirm/original-model/models/model-' + dt_string + '-.h5'
torch.save(model, PATH)