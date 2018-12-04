import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.functional as F
from PIL import Image
import json
import os
import sys 

num_classes = 4

checkpoint = torch.load("./models/custom_model13.model")
model = resnet18(num_classes=num_classes) 

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

if __name__ == "__main__":
    imagefile = sys.argv[1]
    imagepath = os.path.join(os.getcwd(),imagefile)
    prediction = predict_image(imagepath)
    print("Predicted Class: ",prediction)
