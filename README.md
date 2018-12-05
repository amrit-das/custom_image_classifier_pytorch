# Custom Object Detection API PyTorch
Creating machine learning models, for object detection remains a challenge till date. This API is an opensource famework built with the help of PyTorch Framework. The API can be used for training models based on custom datasets. Its a ready to deploy platform to provide custom image training sets as input to a ResNet18 based transfer learning approach. Contributions are most Welcome as this repository is still under building. We are trying to add more and more features and make the work versatile. 

# Usage
For training, put the training data in Dataset/train and testing data in Dataset/val and run the following code
```bash
python main.py 
```
In order to run the trained model, change the path to the model in inference.py and run:
```bash
python inference.py image_filename
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
