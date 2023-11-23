############################################################################################################
###	  									trainer.py
###		Written by Ethan Cannaert Nov 2023
###		Acts as the container for all the ingredients necessary for CNN training
###		
###		
###		Inputs: input dataset X and corresponding labels y, optional seed and batch size 
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
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import cv2
class trainer:
	def __init__(self, X, y, SEED=96, batch_size=4):
		self.SEED = SEED
		self.batch_size = batch_size
		self.X = X
		self.y = y
		self.set_seeds(self.SEED)
		self.device = self.set_device()		

		### shuffle and then split into training and validation (=test) sets
		zipped_Xy = list(zip(self.X, self.y))
		random.shuffle(zipped_Xy)
		self.X, self.y = zip(*zipped_Xy)
		self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=0.15, random_state=SEED)
		self.train_dataset    = ImageData.ImageData(self.xtrain,self.ytrain)
		self.validate_dataset = ImageData.ImageData(self.xtest,self.ytest)

		self.trainloader, self.testloader = self.set_loaders()
		self.model = CustomNN.CustomNN().to(self.device)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.model.parameters(), lr=0.001,momentum = 0.9) ### stochastic gradient descent
		print("Finished initializing training variables.")

	def set_seeds(self,seed_):
		np.random.seed(seed_)
		random.seed(seed_)
		torch.cuda.manual_seed(seed_)
		torch.cuda.manual_seed_all(seed_)
		torch.manual_seed(seed_)
		torch.backends.cudnn.benchmark = True
		print("Seed set with SEED=%i"%seed_)
		### will these be only be set in the function scope?
		return

	def set_device(self):
		if torch.cuda.is_available():
			device = ('cuda:0')
			print("Found CUDA:0 - using CUDA.") ### CUDA = API for interfacing with GPU, (Compute Unified Device Architecture)
		else:
			device = ('cpu')
			print("CUDA not found - using CPU.")
		return device

	def set_loaders(self):
		### init data loaders

		batch_size = 4 
		### now load data into ptytorch
		trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
		                                          shuffle=True, num_workers=2)
		testloader = torch.utils.data.DataLoader(self.validate_dataset, batch_size=self.batch_size,
		                                          shuffle=True, num_workers=2)
		return trainloader, testloader
	def train_epoch(self):
		print('Training')
		self.model.train()
		running_loss = 0.0
		running_correct = 0.0
		for i, data in enumerate(self.trainloader):
			data, target = data[0].to(self.device), data[1].to(self.device)
			self.optimizer.zero_grad()
			outputs = self.model(data)
			loss = self.criterion(outputs, target)
			running_loss += loss.item()  ## batch loss / number of items in batch
			_, preds = torch.max(outputs.data, 1)   ### max value of input tensor
			running_correct += (preds == target).sum().item()
			loss.backward()
			self.optimizer.step()
			
		train_loss = running_loss/len(self.trainloader.dataset)
		train_acc = 100. * running_correct/len(self.trainloader.dataset)
		print("Training loss is %f, accuracy is %f"%(np.around(train_loss),np.around(train_acc)))
		return train_acc, train_loss
	def val_epoch(self):
		print('Validating')
		self.model.eval()
		running_loss = 0.0
		running_correct = 0.0
		with torch.no_grad():
			for i, data in enumerate(self.testloader):
				data, target = data[0].to(self.device), data[1].to(self.device)
				outputs = self.model(data)
				loss = self.criterion(outputs, target)
				
				running_loss += loss.item()
				_, preds = torch.max(outputs.data, 1)
				running_correct += (preds == target).sum().item()
			
			val_loss = running_loss/len(self.testloader.dataset)
			val_acc = 100. * running_correct/len(self.testloader.dataset)
		print("Validation loss is %f, accuracy is %f"%(np.around(val_loss),np.around(val_acc)))
			
		return val_acc, val_loss

	def train(self):

		train_loss =[]
		train_acc = []
		val_loss = []
		val_acc = []
		start_time = time.time()
		for epoch in range(10):  # loop over the dataset multiple times
			print("Starting epoch %i"%epoch)
			_train_acc, _train_loss = self.train_epoch()
			_val_acc, _val_loss = self.val_epoch()
			train_acc.append(_train_acc)
			train_loss.append(_train_loss)
			val_acc.append(_val_acc)
			val_loss.append(_val_loss)
		print("train_loss/train_acc/val_loss/val_acc = ", train_loss,train_acc,val_loss,val_acc)
		self.make_plots(train_loss,train_acc,val_loss,val_acc)
		print('Training and validation are done. Took %f seconds'%(np.around(time.time()-start_time,3)))
		print("Saving model to %s"%(os.path.join('%s'%os.sep.join(os.path.dirname(__file__).split(os.sep)[:-2] ))+"/models"))
		torch.save(self.model.state_dict(), os.path.join('%s'%os.sep.join(os.path.dirname(__file__).split(os.sep)[:-2] ))+"/models/test_model.pth")
		return
	def make_plots(self,train_loss,train_acc,val_loss,val_acc):

		plot_base_dir = os.path.join('%s'%os.sep.join(os.path.dirname(__file__).split(os.sep)[:-2] ))+"/plots/"
	 	# accuracy 
		plt.plot(train_acc, color='yellow', label='train acc')
		plt.plot(val_acc, color='blue', label='validataion acc')
		plt.xlabel('epochs')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(plot_base_dir+"test_accuracy.png")
		 
		# loss plots
		plt.plot(train_loss, color='orange', label='train loss')
		plt.plot(val_loss, color='red', label='val loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(plot_base_dir+"test_loss.png")
		print("Saved accuracy and loss plots at %s"%plot_base_dir)
		return

