# Custom Training PyTorch

The above code can be used/modified for Custom Dataset training and creation of Model in PyTorch.
The code is still in beta-testing. Open for pull requests. 

# Usage
For training, put the training data in Dataset/train and testing data in Dataset/val and run the following code
```bash
python main.py 
```
In order to run the trained model, change the path to the model in inference.py and run:
```bash
python inference.py
```

Required Libraries:
  PyTorch - 0.4.1
  Cuda - Optional (For faster training)

#Installing requirements
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
