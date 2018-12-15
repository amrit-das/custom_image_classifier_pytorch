# PyTorch Based Custom Image Classifier

Creating machine learning models, for Image Clasification, built with the help of [PyTorch](http://pytorch.org) Framework. The API can be used for training models based on custom datasets. Its a ready to deploy platform to provide custom image training sets as input to a ResNet18 based transfer learning approach. Contributions are most Welcome as this repository is still under building. We are trying to add more and more features and make the work versatile. 

##### Dataset : Contains the training and testing datasets

##### models : Contains the trained models/checkpoints

##### Predict Image : This folder is used to store the images/videoes to be predicted/segmented

##### train.py : To train the data

##### predict.py : To predict from the trained model

##### hyper_params.json : The json file contains all the hyper patameters and related notations. 

### Table of Contents
- <a href='#Installation'> Installation </a>
- <a href='#Usage'> Usage </a>
- <a href='#Version'> Version </a>
- <a href='#todo'> Future Work </a>
- <a href='#reference'> Reference </a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation

- PyTorch can be installed from the website [pytorch.org](https://pytorch.org)

   *If installing for ARM devices, follow the [Link](https://medium.com/hardware-interfacing/how-to-install-pytorch-v4-0-on-raspberry-pi-3b-odroids-and-other-arm-based-devices-91d62f2933c7)
- Example dataset can be downloaded from this [Link](https://download.pytorch.org/tutorial/hymenoptera_data.zip)
- Clone the repository and Install requirements 
```bash
sudo apt install git
git clone https://github.com/amrit-das/custom_image_classifier_pytorch.git
cd custom_image_classifier_pytorch
sudo pip install -r requirements.txt
```
## Usage
You can check the detailed usage @ https://medium.com/hardware-interfacing/custom-image-classifier-using-transfer-learning-in-pytorch-framework-c2f7f5155239
### Hyperparameters
Before training the network, you can tune the network to your hyper parameters using the *'hyper_params.json'* script. But it is advisable not to change any parameters that you may not be aware of, doing so might mess up the network.

### Training 
Once modified the hyperparameters, put the training data in Dataset/train and testing data in Dataset/val and run the following code

```bash
python train.py
```
### Predicting
In order to predict from trained model, place your image to be predicted in Predict_Image and run:
```bash
python predict.py -i image_name_to_be_predicted -m model_name -n num_of_classes 
```
## Version

PyTorch - 0.4.1

NumPy - 1.15.4

OpenCV -  3.4.4

Cuda - Optional 

## ToDo
- Add Tutorials
- Add Support for Video Predictions
- Add Folder Segregation ( In progress )
- Add Object Detection
- Bounding Boxes


## Reference

PyTorch Tutorials - ( https://pytorch.org/tutorials/ )

Medium Blogs - ( https://medium.com/hardware-interfacing/ )
