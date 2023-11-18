import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import normalize
from PIL import Image
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


###########################################################################################
###  jpg_to_tensor_converter.py
###  Written by Ethan Cannaert, Nov 2023
###  Takes no inputs, convers jpgs in the train and test datasets
###	 and writes out .pt files of tensors representing each of these jpgs
### NOTE: This still crashes on this laptop, so it's a RAM issue, not a jupyter issue
###########################################################################################
### class to load and store data
class image_data:   
	def __init__(self):
		self.label_converter = dict()
		self.convert_label_to_int()
		self.train_data	= [] # train data in tensor format
		self.test_data	 = [] # test data in tensor format
		self.validate_data = [] # validation data in tensor format
		self.train_labels =  []
		self.test_labels  =  []
		self.image_files = dict()
		self.load_train_data()
		self.load_test_data()

		torch.save(self.train_data, 'train_tensors.pt')
		torch.save(self.train_labels, 'train_tensors.pt')
		torch.save(self.test_data, 'test_tensors.pt')
		torch.save(self.test_labels, 'test_tensors.pt')

	### helper function to convert str letters/del/space into numbers 
	def convert_label_to_int(self):
		alphabet = "A/B/C/D/E/F/G/H/I/J/K/L/M/N/O/P/Q/R/S/T/U/V/W/X/Y/Z/del/nothing/space"
		for iii,label in enumerate(alphabet.split("/")):
			self.label_converter[label] = iii
	### give this an image filename and it will return the transformed tensor 
	def convert_jpg_to_tensor(self,filepath):
		"""
		image = Image.open(filepath) 
		# image to a Torch tensor 
		transform = transforms.Compose([ 
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])  """ 
		#transform = transforms.PILToTensor() 
		### there is an integrated method for this that is much faster, but it doesn't normalize
		return torchvision.io.read_image(filepath)
		#return transform(image)
	### load training data into the instance variable train_data
	def load_train_data(self):
		now = time.time()
		train_path = "datasets/asl_alphabet_train/asl_alphabet_train/"
		train_directories = glob.glob(train_path+"/*")
		n_files = 0
		print(" ----- Loading training dataset -----")
		"""transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])"""
		for dir_ in train_directories:
			letter = dir_.split("/")[-1]
			self.image_files[letter] = []
			n_per_letter = 0
			for image_file in glob.glob(dir_+"/*"):
				if (n_per_letter > 1200):
					continue
				#image = Image.open(image_file) 
				#self.train_data.append( transform(image))
				self.image_files[letter].append(image_file)
				self.train_data.append(self.convert_jpg_to_tensor(image_file) )
				self.train_labels.append(self.label_converter[letter])
				n_per_letter+=1
				n_files +=1
			print("Finished importing %s"%letter)
		print("Done with training dataset - converted %i files. Took %f seconds"%(n_files, np.around(time.time()-now)))
		return
	### load test data into the instance variable train_data
	def load_test_data(self):
		now = time.time()
		test_path = "datasets/asl_alphabet_test/asl_alphabet_test/"
		test_directories = glob.glob(test_path+"/*")
		n_files = 0
		print("----- Loading test dataset -----")
		"""transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])"""
		for image_file in test_directories:
			letter = image_file.split("_")[-2].split("/")[-1]
			self.image_files[letter] = []
			self.image_files[letter].append(image_file)

			#image = Image.open(image_file) 
			#self.train_data.append( transform(image))

			self.test_data.append(self.convert_jpg_to_tensor(image_file))
			self.test_labels.append(self.label_converter[letter])
			n_files +=1
		print("Done with test dataset - converted %i files. Took %f seconds"%(n_files, np.around(time.time()-now,4)))
		return

	


if __name__=="__main__":
	id_ = image_data()