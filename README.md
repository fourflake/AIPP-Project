# Project: Image Classifier Command Line Application
Code written for Udacity's AI Programming With Python Nanodegree project.

An image classifier is built with PyTorch and converted into a command line application. The functionality includes training a neural network on any new classified image data set, specifying which network architecture to use (currently implemented for 'vgg16', 'alexnet', 'densenet161') and saving the trained network to a chosen checkpoint. With a trained network, it's possible to load this checkpoint and make predictions on input images and choose the amount of most likely classes to show. 

# Installation

Clone the GitHub repository and make sure python 3.6 (or later) and Numpy,Pandas,Seaborn packages are installed.

Install pytorch (see https://pytorch.org/ for system specific installation commands):

`conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

# Usage

- Images
  -  Make sure to have classified images stored in a seperated 'train', 'valid' and 'test' folder.
  - In the main folder (e.g.'flowers'), there exist 3 subfolders ('train','valid' and 'test) that each contain 101 class folders (e.g. '1','2',...'101'). Each of these class folders contain a certain amount of images of that category of flowers. 
  - For inspiration for datasets to use see http://deeplearning.net/datasets/

- Training
  - Trains a neural network with images located in a specific folder (containing the subfolders as explained above). This function prints intermediary losses and accuracies and saves the final model at a specified checkpoint.
  - Example function call: Train using the 'Alexnet' architecture, with a classifier of 2 hidden layers of 500 units each, for 4 epochs, using the gpu if availabe. 
  
  `python train.py '/c/Users/Joeri/GitHub/flowers' --arch alexnet --epochs 4 --hidden_units 500 --gpu --save_dir`

- Predicting
  - Predicts the class for an input image by using a trained network that is being loaded from a specified checkpoint.
  - Example function call: Predict the top 3 most likely classes for a given input image, using the gpu for inference if available
  
  `python predict.py "flowers/test/1/image_06743.jpg" checkpoint_1.pth --top_k 3 --gpu`
