"""
Image Classifier Project.
Written by Joeri Jessen 
Date: 15/04/2020

utility_functions.py contains all necessary utility functions for the image classifier project.
"""

import torch
from torchvision import datasets, transforms, models
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import os


def data_transforms(size,median_list,std_list):
    """ Function returns a dictionary with the transforms for train,validation and testing dataset
    
    Parameters:
        size (int): Required output image size
        median_list (lst): Required median list for 3 dim RGB normalization
        std_list (lst): Required standard deviation list for 3 dim RGB normalization
    
    Returns:
        data_transforms (dict) : Dictionary with keys:
                                                   'train': transforms for training. utilizing data augmentation
                                                   'valid': transforms for validation
                                                    'test': transforms for testing data
                                                    
    """
    
    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(size),
                                             transforms.RandomRotation(30),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(median_list,
                                                                  std_list)]),

        'valid': transforms.Compose([transforms.Resize((size,size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(median_list,
                                                                   std_list)]),
        'test': transforms.Compose([transforms.Resize((size,size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(median_list,
                                                                  std_list)])
    }
    return data_transforms


# Load the datasets with ImageFolder
def load_datasets(data_dir,transforms):
    
    """ At a given directory that has seperated folders for 'train,'valid' and 'test' data, 
    
    Returns
    Tuple: (dictionary with images seperated by folder, number of classes in data)
    
    """ 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    image_datasets = {
                'train' : datasets.ImageFolder(train_dir, transform=transforms['train']),

                'valid' : datasets.ImageFolder(valid_dir, transform=transforms['valid']),

                'test' : datasets.ImageFolder(test_dir, transform=transforms['test'])
                 }
    n_classes = len(os.listdir(train_dir))
    return image_datasets, n_classes


# Using the image datasets and the trainforms, define the dataloaders
def create_dataloaders(image_datasets,batch):
    """  Create dataloader from image datasets, having a certain batch size"""
    dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'],batch_size=batch,shuffle=True),
        
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'],batch_size=batch,shuffle=True),
        
        'test' : torch.utils.data.DataLoader(image_datasets['test'],batch_size=batch)
    }
    return dataloaders


def save_checkpoint(model, save_dir='checkpoint_default.pth'):

    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'optimizer' : model.optimizer,
                  'optimizer_state_dict' : model.optimizer.state_dict(),
                  'epochs' : model.epochs,
                  'arch' : model.arch,
                  'classifier' : model.classifier}

    torch.save(checkpoint, save_dir)
    print('Model checkpoint saved at: ',save_dir)

    
def load_checkpoint(checkpoint_path): 
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['arch'](pretrained=True)

    for param in model.parameters():
            param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']

    optimizer = model.optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    #model.to(device)
    return model, optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_im = Image.open(image)
    pil_im = pil_im.resize((256,256))
    
    width, height = pil_im.size   # Get dimensions
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    pil_im = pil_im.crop((left, top, right, bottom))
    
    np_image = np.array(pil_im)
    np_image_float = np_image/255
    np_image_normalized = (np_image_float-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    
    np_changed_order = np_image_normalized.transpose(2,0,1)
    
    tensor_out = torch.from_numpy(np_changed_order)

    return tensor_out.float()


def load_class_mapping(category_names = 'cat_to_name.json'):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    

