"""
Image Classifier Project.
Written by Joeri Jessen 
Date: 15/04/2020

model_functions.py contains all necessary functions to build and train a network on a dataset,
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
import utility_functions as utility

def build(n_classes, arch='vgg16',hidden_units=400,gpu=False):

    device = torch.device("cuda" if (gpu==True and torch.cuda.is_available()) else "cpu")
   
    arch_attr = getattr(models, arch)
    model = arch_attr(pretrained=True)
    model.arch = arch_attr

    for param in model.parameters():
        param.requires_grad = False

    if type(model.classifier) == torch.nn.modules.linear.Linear:
        input_units = model.classifier.in_features
    elif type(model.classifier[0]) == torch.nn.modules.linear.Linear:
        input_units = model.classifier[0].in_features
    else:
        input_units = model.classifier[1].in_features
        
    model.classifier = nn.Sequential(nn.Linear(input_units,hidden_units),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(hidden_units,hidden_units),
                                nn.ReLU(),     
                                nn.Linear(hidden_units,n_classes),
                                nn.LogSoftmax(dim=1))
    model.to(device)
    
    return model


def train(model,dataloaders,print_every = 4,learning_rate=0.001,gpu=False,epochs=3):
    '''Function to train a given model with certain hyper parameters.
    Prints out training loss, validation loss, and validation accuracy as the network trains
    '''
    device = torch.device("cuda" if (gpu==True and torch.cuda.is_available()) else "cpu")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),learning_rate)

    steps= 0 
    model.epochs = epochs

    with active_session():
        for e in range(epochs):
            train_loss = 0
            for images, labels in dataloaders['train']:
                steps += 1
                images, labels = images.to(device), labels.to(device)

                logps = model(images)
                loss = criterion(logps,labels)
                ps = torch.exp(logps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                
                if steps % print_every == 0:
                    accuracy = 0
                    valid_loss = 0
                    model.eval()

                    with torch.no_grad():
                        for images,labels in dataloaders['valid']:

                            images, labels = images.to(device), labels.to(device)

                            logps = model(images)
                            loss = criterion(logps,labels)
                            ps = torch.exp(logps)

                            top_p,top_cl = ps.topk(1,dim=1)

                            equals = top_cl == labels.view(*top_cl.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))
                            valid_loss += loss

                        print(f"Epoch: {e+1}/{epochs} .. "
                              f"Train Loss: {train_loss/len(dataloaders['train']):.3} .. "
                              f"Test Loss: {valid_loss/len(dataloaders['valid']):.3} .. " 
                              f"Accuracy: {accuracy/len(dataloaders['valid']):.3}")
                    model.train()     
            print("Epoch completed")  
        print("Training model completed.")   
    model.optimizer = optimizer
    return model 
    
    
def predict(image_path, model, top_k=1, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if (gpu==True and torch.cuda.is_available()) else "cpu")
    model.to(device)

    im = utility.process_image(image_path)
    im_in = im.view(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        model.eval()
        logps = model(im_in)
        ps = torch.exp(logps)

    probs, idx_list = ps.topk(top_k,dim=1)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_list =[]
    [class_list.append(idx_to_class[idx]) for idx in idx_list.cpu().numpy().tolist()[0]]
    
    return probs.cpu().numpy()[0], class_list
    



