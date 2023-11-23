############################################################################################################
###	  									training.py
###		Written by Ethan Cannaert Nov 2023
###		Creates an instance of the ImageLoadData, LoadData, and CustomNN methods in order to set up the 
###		loss function, validation, and training loops to execute this NN
###		
###		Inputs: int = number of data files to load per category. Ex: python3 training.py 1000
############################################################################################################


import random
import sys, os
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as transform
from sklearn.model_selection import train_test_split
from LoadImageData import LoadImageData
from ImageData import ImageData
from CustomNN import CustomNN
import numpy as np
import pandas as pd
import cv2
from trainer import trainer
import time

def main(args):

	base_dir = os.path.join('%s'%os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1] )) ## wonder if there is an easier way to get this ...
	if len(args) != 2:
		print("Incorrect inputs. Use format python3 make_csv.py <int = number of files of each category to load>")
		return

	### Establish seeds for reproducibility 
	SEED = 96
	n_files = int(args[1])
	datasets = LoadImageData.LoadImageData(n_files)

	### get images from LoadImageData class
	X = datasets.get_train_data()
	y = datasets.get_train_labels_num()

	### define trainer class that holds the relevant training objects
	trainer_ = trainer.trainer(X,y,SEED)
	trainer_.train()

if __name__=="__main__":
	main(sys.argv)
