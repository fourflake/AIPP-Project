"""
Image Classifier Project.
Written by Joeri Jessen 
Date: 16/04/2020

predict.py uses a trained network to predict the class for an input image.
"""
import utility_functions as utility
import model_functions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="The path of the input image to make a class prediction on",type=str)
parser.add_argument("checkpoint", help="The path of the saved checkpoint of the network to use for prediction",type=str)
parser.add_argument("--top_k", help="Optionally set amount of top K most likely classes to return",type=int)
parser.add_argument("--category_names", help="Optionally use a json mapping of categories to real names", type=str)
parser.add_argument("--gpu", help="Optionally use GPU for inference", action="store_true")

args = vars(parser.parse_args())
args_dict = dict(filter(lambda elem: (elem[1] != None) and (elem[1] != False), args.items()))

kwargs_mapping =  {k:v for (k,v) in args_dict.items() if k in ['category_names']}
cat_to_name = utility.load_class_mapping(**kwargs_mapping)

model, optimizer = utility.load_checkpoint(args_dict['checkpoint'])

kwargs_predict =  {k:v for (k,v) in args_dict.items() if k in ['top_k','gpu']}
probs, classes = model_functions.predict(args_dict['image_path'], model, **kwargs_predict)

classes_names = [cat_to_name[cl] for cl in classes]
predictions = zip(classes_names,probs)
print("Prediction:")
[print("{} with a probability of {:.3}".format(cl,p)) for cl,p in predictions]


## Test: python predict.py "flowers/test/1/image_06743.jpg" checkpoint_1.pth --top_k 3


