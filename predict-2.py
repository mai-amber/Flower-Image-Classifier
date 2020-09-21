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


parser = argparse.ArgumentParser()

parser.add_argument('--json_file', type=str, default='cat_to_name.json')
parser.add_argument('--image_path', type=str, default='./flowers/train/42/image_05697.jpg')
parser.add_argument('--filepath', type=str, default='checkpoint.pth')
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--gpu', default='gpu', type=str)

classifier_inputs = parser.parse_args()

json_file = classifier_inputs.json_file
image_path = classifier_inputs.image_path
filepath = classifier_inputs.filepath
topk = classifier_inputs.topk
gpu = classifier_inputs.gpu


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    gpu = checkpoint['gpu']
    hidden_layer = checkpoint['hidden_layer']
    lr = checkpoint['learning_rate']
    gpu = checkpoint['gpu']
    epochs = checkpoint['epochs']
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
    

model = load_checkpoint(filepath)

def process_image(image_path):

    image = Image.open(image_path)
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    np_array = transform(image).float()
    
    return np_array


def predict(image_path, model, topk=5):
    
    images = process_image(image_path)
    images = images.unsqueeze(0)
    
    if gpu == 'gpu': model.to('cuda:0')
        
    with torch.no_grad():
        if gpu == 'gpu': images = images.to('cuda')
        outputs = model.forward(images)
        
    prediction = F.softmax(outputs.data, dim = 1)
    
    probs, indices = prediction.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    
    return probs, classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

probs, classes = predict(image_path, load_checkpoint(filepath), topk)
flower_names = [cat_to_name[str(i)] for i in classes]
for result in zip(flower_names, probs):
    print(result)