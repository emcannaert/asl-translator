##############################################
###	  LoadImageData.py				  ###
###  Written by Ethan Cannaert, Nov 2023   ###
###  Defines LoadImageData class that	  ###
###  loads and stores train/test info.	 ###
###  Inputs: int = number of files of each ###
###  category to load. Default = 3000	  ###
##############################################

import torch
import torchvision
import numpy as np
import time
import glob
import os

class LoadImageData:   

	### takes in the number of images from each category to load and loads test and train data and labels
	def __init__(self, n_files_per_class=3000):  ### n_files == number of files of each letter to train on
		self.label_converter = dict()
		self.convert_label_to_int()
		self.base_dir = os.path.join('%s'%os.sep.join(os.path.dirname(__file__).split(os.sep)[:-2] ))
		self.n_files_per_class = n_files_per_class
		self._train_data	= [] # train data paths
		self._test_data	 = [] # test data paths
		self._train_labels = [] # truth LETTER label for training data
		self._test_labels   = [] # truth LETTER label for test data
		self._train_labels_num =  [] # truth NUMBER label for training data
		self._test_labels_num  =  [] # truth NUMBER label for test data
		
		self.image_path_dict = dict() # dictionary of all training file paths
		self.load_train_data()
		self.load_test_data()
	  	
	### helper function to convert str letters/del/space into numbers 
	def convert_label_to_int(self):
		alphabet = "A/B/C/D/E/F/G/H/I/J/K/L/M/N/O/P/Q/R/S/T/U/V/W/X/Y/Z/del/nothing/space"
		for iii,label in enumerate(alphabet.split("/")):
			self.label_converter[label] = iii
  
	### load training data and labels into the instance variables _train_data and _train_labels
	def load_train_data(self):
		now = time.time()
		train_path = self.base_dir+ "/datasets/asl_alphabet_train/asl_alphabet_train/"

		train_directories = glob.glob(train_path+"/*")
		n_files = 0
		print(" ----- Loading training dataset -----")

		for dir in train_directories:
			letter = dir.split("/")[-1]
			self.image_path_dict[letter] = []
			n_test_for_class = 0 
			for image_file in glob.glob(dir+"/*"):
				if n_test_for_class > (self.n_files_per_class-1):
					break ### move onto the next letter 
				self.image_path_dict[letter].append(image_file)
				self._train_data.append( image_file )
				self._train_labels_num.append(self.label_converter[letter])
				self._train_labels.append(letter)
				n_test_for_class+=1
				n_files +=1
			#print("Finished importing %s"%letter)
		print("Done importing training dataset - loaded paths for %i files. Took %f seconds"%(n_files, np.around(time.time()-now)))
		return
	### load test data and labels into the instance variable _test_labels and _test_data
	def load_test_data(self):
		now = time.time()
		test_path = self.base_dir+ "/datasets/asl_alphabet_test/asl_alphabet_test/"
		test_directories = glob.glob(test_path+"/*")
		n_files = 0
		print("----- Loading test dataset -----")
		for image_file in test_directories:
			letter = image_file.split("_")[-2].split("/")[-1]
			self.image_path_dict[letter] = []
			self.image_path_dict[letter].append(image_file)
			self._test_data.append(image_file   )
			self._test_labels_num.append(self.label_converter[letter])
			self._test_labels.append(letter)
			n_files +=1
		print("Done with test dataset - loaded paths for %i files. Took %f seconds"%(n_files, np.around(time.time()-now,4)))
		return

	### getters and setters
	def get_test_data(self):
		return self._test_data
	def get_train_data(self):
		return self._train_data
	def get_test_labels(self):
		return self._test_labels
	def get_train_labels(self):
		return self._train_labels
	def get_train_labels_num(self):
		return self._train_labels_num

	def set_test_data(self,new_test_data):
		self._test_data = new_test_data
		return 
	def set_train_data(self,new_train_data):
		self._train_data = new_train_data
		return 
	def set_test_labels(self,new_test_labels):
		self._test_labels = new_test_labels
		return 
	def set_train_labels(self, new_train_labels):
		self._train_labels = new_train_labels
		return 







