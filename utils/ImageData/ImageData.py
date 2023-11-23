##############################################
###		  ImageData.py				       ###
###  Written by Ethan Cannaert, Nov 2023   ###
###  Defines ImageData class that		   ###
###  stores train/test information.		   ###
###  Transformed tensors for each data	   ###
###  element can be accessed with [].	   ###
###  Inputs: (data,labels)				   ###
##############################################
import torch
import torchvision
import numpy as np
import albumentations
import cv2
class ImageData:
	### take in image paths and category labels
	def __init__(self,paths, labels):
		self.X = paths
		self.y = labels
		self.transf = albumentations.Compose([
			albumentations.Resize(224, 224, always_apply=True)])
	### the built-in len() method is necessary to give this to the dataloaders
	def __len__(self):
		return (len(self.X))
		
	### Built-in method to access each tensor element with []. EX: ImageData[23]
	### Note: each tensor is resized to 224x224, a standard size for CNNs
	def __getitem__(self,i):
		image = cv2.imread(self.X[i])
		image = self.transf(image=np.array(image))['image']
		image = np.transpose(image, (2, 0, 1)).astype(np.float32)
		label = self.y[i]
		return torch.tensor(image, dtype=torch.float), torch.tensor(label, dtype=torch.long)