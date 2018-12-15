# PyTorch Based Custom Image Classifier

Creating machine learning models, for Image Clasification, built with the help of PyTorch Framework. The API can be used for training models based on custom datasets. Its a ready to deploy platform to provide custom image training sets as input to a ResNet18 based transfer learning approach. Contributions are most Welcome as this repository is still under building. We are trying to add more and more features and make the work versatile. 

Dataset : Contains the training and testing datasets

models : Contains the trained models/checkpoints

Predict Image : This folder is used to store the images/videoes to be predicted/segmented

main.py : To train the data

predict.py : To predict from the trained model

hyper_params.json : The json file contains all the hyper patameters and related notations. 

# Usage
For training, put the training data in Dataset/train and testing data in Dataset/val and run the following code

# Before training the network, you can tune the network to your hyper parameters using the 'hyper_params.json' script. But it is advisable not to change any parameters that you may not be aware of, doing so might mess up the network.

```bash
python main.py 
```
In order to predict from trained model, place your image to be predicted in /Predict_Image and run:
```bash
python predict.py -i image_name_to_be_predicted -m model_name -n num_of_classes 
```

# Required Packages:

PyTorch - 0.4.1

NumPy - 1.15.4

OpenCV -  3.4.4

Cuda - Optional 

# References

PyTorch Tutorials - ( https://pytorch.org/tutorials/ )

Medium Blogs
