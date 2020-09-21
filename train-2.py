import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image
import torchvision.models
import json
import argparse
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default="./flowers/")
parser.add_argument('--save_dir', type=str, default='./checkpoint.pth')
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--hidden_layer', type=int, default=4096)
parser.add_argument('--learning_rate', type=float, default=.0001)
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--gpu', type=str, default='gpu')

classifier_inputs = parser.parse_args()

data_dir = classifier_inputs.data_dir
save_dir = classifier_inputs.save_dir
epochs = classifier_inputs.epochs
hidden_layer = classifier_inputs.hidden_layer
lr = classifier_inputs.learning_rate
arch = classifier_inputs.arch
gpu = classifier_inputs.gpu

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
testing_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
validation_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
testing_data = datasets.ImageFolder(test_dir, transform=testing_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

trainloaders = torch.utils.data.DataLoader(training_data, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(testing_data, batch_size = 64)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size = 64)


arch = 'vgg16'
lr = .0001
hidden_layer = 4096
gpu = 'gpu'
epochs = 6

if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_size = 25088
elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        input_size = 1024
if gpu == 'gpu':
        model.to('cuda')
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(25088, 4096)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(4096, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

epochs = 6
steps = 0
running_loss = 0
print_every = 60

if gpu == 'gpu':
    model.to('cuda')

for epoch in range(epochs):
    for images, labels in trainloaders:
        steps += 1
        
        if gpu == 'gpu':
            images, labels = images.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if steps % print_every == 0:
            accuracy = 0
            valid_loss = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validationloader:
                    if gpu == 'gpu':
                        images, labels = images.to('cuda'), labels.to('cuda')
                    outputs = model.forward(images)
                    loss = criterion(outputs, labels)
                    
                    valid_loss += loss.item()

                    ps = torch.exp(outputs)
                    top_p, top_class = ps.topk(1,dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validationloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()
            
accuracy = 0
test_loss = 0
model.eval()

if gpu == 'gpu':
        model.to('cuda')
        
with torch.no_grad():
    for images, labels in testloader:
        if gpu == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
                    
        test_loss += loss.item()

        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1,dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
print(f"Test loss: {test_loss/len(testloader):.3f}.. "
f"Test accuracy: {accuracy/len(testloader):.3f}")

model.class_to_idx = training_data.class_to_idx
checkpoint = {'input_size': 25088,
             'output_size': 102,
             'arch': arch,
             'gpu': gpu,
             'epochs': 6,
             'hidden_layer': 4096,
             'learning_rate': 0.0001,
             'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'model_state_dict': model.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')
