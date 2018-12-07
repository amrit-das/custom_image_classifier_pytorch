import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.functional as F
from PIL import Image
import os
import sys
import argparse

parser = argparse.ArgumentParser(description = 'To Predict from a trained model')
parser.add_argument('-i','--image', dest = 'image_name', required = True, help='Path to the image file')
parser.add_argument('-m','--model', dest = 'model_name', required = True, help='Path to the model')
parser.add_argument('-n','--num_class',dest = 'num_classes', required = True, help='Number of training classes')
args = parser.parse_args()

path_to_model = "./models/"+args.model_name
checkpoint = torch.load(path_to_model)

model = resnet18(num_classes = int(args.num_classes))
model.load_state_dict(checkpoint)
model.eval()

def predict_image(image_path):
    print("prediciton in progress")
    image = Image.open(image_path)
    transformation = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image_tensor = transformation(image).float()
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    input = Variable(image_tensor)
    output = model(input)

    index = output.data.numpy().argmax()
    return index

def class_mapping(index):
    mapping=open('class_mapping.txt','r')
    class_map={}
    for line in mapping:
        l=line.strip('\n').split('~')
        class_map[l[1]]=l[0]
    return class_map[str(index)]

if __name__ == "__main__":

    imagepath = "./Predict_Image/"+args.image_name
    prediction = predict_image(imagepath)
    name = class_mapping(prediction)
    print("Predicted Class: ",name)
