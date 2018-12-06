# Custom Image Training and Prediction API PyTorch
Creating machine learning models, for object detection remains a challenge till date. This API is an opensource famework built with the help of PyTorch Framework. The API can be used for training models based on custom datasets. Its a ready to deploy platform to provide custom image training sets as input to a ResNet18 based transfer learning approach. Contributions are most Welcome as this repository is still under building. We are trying to add more and more features and make the work versatile. 

# Usage
For training, put the training data in Dataset/train and testing data in Dataset/val and run the following code
```bash
python main.py 
```
In order to predict from trained model, place your image to be predicted in /Predict_Image and run:
```bash
python inference.py -i image_name_to_be_predicted -m model_name -n num_of_classes 
```

Required Libraries:

PyTorch - 0.4.1

Cuda - Optional (For faster training)

# Installing requirements
Linux
```bash
sudo pip install torch torchvision
```
Mac
```bash
sudo pip install torch torchvision
```
Windows

PyTorch doesnot support Python 2.7 on Windows
```bash
sudo pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp35-cp35m-win_amd64.whl
pip3 install torchvision
``` 
# References

PyTorch Tutorials - (https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

Medium Blogs
