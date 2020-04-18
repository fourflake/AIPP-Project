"""
Image Classifier Project.
Written by Joeri Jessen 
Date: 15/04/2020

Train.py will train a new network on a dataset and save the model as a checkpoint

Prints out training loss, validation loss, and validation accuracy as the network trains
"""

import utility_functions as utility
import model_functions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="The path of data directory to train the network on",type=str)
parser.add_argument("--save_dir", help="Optionally set the directory to save checkpoints",type=str)
parser.add_argument("--arch", help="Choose model architecture",choices=['vgg16','alexnet','densenet161'], type=str)
parser.add_argument("--learning_rate", help="Optionally set a learning rate for the model",type=float)
parser.add_argument("--hidden_units", help="Optionally set an amount of hidden units in the architecture",type=int)
parser.add_argument("--epochs", help="Optionally set the amount of epochs to train",type=int)
parser.add_argument("--gpu", help="Optionally use GPU for training", action="store_true")

args = vars(parser.parse_args())
args_dict = dict(filter(lambda elem: (elem[1] != None) and (elem[1] != False), args.items()))

# Define transforms for training, validation and test data, given the required model input image size and normalization
transforms = utility.data_transforms(224,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

# Load the datasets with ImageFolder, using the parameter transforms 
image_datasets, n_classes = utility.load_datasets(args_dict['data_dir'],transforms)

dataloaders = utility.create_dataloaders(image_datasets,batch = 64)

kwargs_model =  {k:v for (k,v) in args_dict.items() if k in ['arch','hidden_units','gpu']}

model = model_functions.build(n_classes, **kwargs_model)
    
model.class_to_idx = image_datasets['train'].class_to_idx

kwargs_train =  {k:v for (k,v) in args_dict.items() if k in ['epochs','gpu','learning_rate']}
model_trained = model_functions.train(model,dataloaders,print_every = 5,**kwargs_train)

kwargs_save =  {k:v for (k,v) in args_dict.items() if k in ['save_dir']}
utility.save_checkpoint(model_trained,**kwargs_save)

## Test: python train.py 'flowers' --gpu --hidden_units 400 --epochs 3
