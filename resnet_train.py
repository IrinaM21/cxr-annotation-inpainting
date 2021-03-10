import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import datetime

###################
## PREPROCESSING ##
###################

num_workers = 0
batch_size = 128
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

print('defined transforms')

# define datasets:
data = datasets.ImageFolder('./3_class_data')

print('found data')

# get idx for images in each class
covid_idx = []
normal_idx = []
vp_idx = []

for idx, (sample, target) in enumerate(data):
    if target == 0:
      covid_idx.append(idx)
    elif target == 1:
      normal_idx.append(idx)
    else:
      vp_idx.append(idx)
      
print('got idx for images in each class')

# get indices:
np.random.shuffle(covid_idx)
covid_test_idx, covid_val_idx, covid_train_idx = covid_idx[:50], covid_idx[50:200], covid_idx[200:]
print('Initialized COVID indices')

np.random.shuffle(normal_idx)
normal_test_idx, normal_val_idx, normal_train_idx = normal_idx[:50], normal_idx[50:200], normal_idx[200:]
print('Initialized NORMAL indices')

np.random.shuffle(vp_idx)
vp_test_idx, vp_val_idx, vp_train_idx = vp_idx[:50], vp_idx[50:200], vp_idx[200:]
print('Initialized Viral Pneumonia indices')

# create custom dataset class to use train and test transforms on data
class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        sample = self.dataset[idx][0]
        # sample = transforms.ToPILImage()(self.dataset[idx][0]).convert("RGB")
        if self.transform:
            sample = self.transform(sample)
        
        target = self.dataset[idx][1]

        return sample, target

# get data for each class from indices
covid_train_data = torch.utils.data.Subset(data, covid_train_idx)
covid_test_data = torch.utils.data.Subset(data, covid_test_idx)
covid_val_data = torch.utils.data.Subset(data, covid_val_idx)
print('got data for COVID')

normal_train_data = torch.utils.data.Subset(data, normal_train_idx)
normal_test_data = torch.utils.data.Subset(data, normal_test_idx)
normal_val_data = torch.utils.data.Subset(data, normal_val_idx)
print('got data for NORMAL')

vp_train_data = torch.utils.data.Subset(data, vp_train_idx)
vp_test_data = torch.utils.data.Subset(data, vp_test_idx)
vp_val_data = torch.utils.data.Subset(data, vp_val_idx)
print('got data for VP')

# concatenate train, test, and val data
train_data = torch.utils.data.ConcatDataset([covid_train_data, normal_train_data, vp_train_data])
test_data = torch.utils.data.ConcatDataset([covid_test_data, normal_test_data, vp_test_data])
val_data = torch.utils.data.ConcatDataset([covid_val_data, normal_val_data, vp_val_data])
print('concatenated train, test, and val data')

# transform data
train_data = CustomDataset(dataset=train_data, transform=train_transform)
test_data = CustomDataset(dataset=test_data, transform=test_transform)
val_data = CustomDataset(dataset=val_data, transform=test_transform)
print('train_data shape:')
print('transformed data')

# initialize data loaders for train, test, and val
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
print('Initialized loaders')


###########
## MODEL ##
###########

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
n_epochs = 90

###########
## TRAIN ##
###########

# lists to keep track of training progress:
train_loss_progress = []
val_accuracy_progress = []

model.train() # prep model for training

n_iterations = int(len(train_data) / batch_size)


for epoch in range(n_epochs):
    
    # monitor training loss
    if not used_checkpts: 
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

    if epoch % 30 == 0:
      scheduler.step()

    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    train_loss_progress.append(train_loss)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

    # Run the test pass:
    correct = 0
    total = 0
    confusion = [[0,0,0],[0,0,0],[0,0,0]]
    model.eval()  # prep model for validation

    with torch.no_grad():
        for data, target in val_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            confusion = np.add(confusion, confusion_matrix(target, predicted, labels=[0,1,2]))
    
    val_acc = (100 * correct / total)
    print('Accuracy of the network on the validation set: %d %%' % (val_acc))
    val_accuracy_progress.append(val_acc)

    print('Confusion matrix for validation set:')
    print(confusion)
    
    ########################
    ## SAVING CHECKPOINTS ##
    ########################

    # appending the date and time to automate renaming of file
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y+%H:%M:%S")

    PATH = './checkpoints/' + dt_string + '.pt'
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, PATH)

#######################
## SAVING THE MODEL ##
#######################

# appending the date and time to automate renaming of file
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")

PATH = './models/model-' + dt_string + '-.h5'
torch.save(model, PATH)

#######################
## TESTING THE MODEL ##
#######################

correct = 0
total = 0
confusion = [[0,0,0],[0,0,0],[0,0,0]]
model.eval()  # prep model for testing

with torch.no_grad():
  for data, target in test_loader:
    outputs = model(data)
    _, predicted = torch.max(outputs.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()
    confusion = np.add(confusion, confusion_matrix(target, predicted, labels=[0,1,2]))

test_acc = (100 * correct / total)
print('Accuracy of the network on the test set: %d %%' % (test_acc))

print('Confusion matrix for test set:')
print(confusion)
