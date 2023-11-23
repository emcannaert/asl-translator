#####################################################################
###								createCSV.py					  ###
###					Written by Ethan Cannaert, Nov 2023		      ###
### 	Creates a CSV file with the training data information and ###  
###		truth labels. The data is shuffled. Both binarized and	  ###
###		non-binarized versions are made, and binarized outputs	  ###
###	 are saved in a pickle file in case they need to be used.     ###
###	 inputs: int = number of files from each category to load     ###
#####################################################################
import numpy as np
import pandas as pd
import pickle
import time
import sys,os
import sklearn.preprocessing
from LoadImageData import LoadImageData

def create_csv(n_files):

	
	base_dir = os.path.join('%s'%os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1] ))

	datasets = LoadImageData.LoadImageData(n_files) ## data instance, passing in 1000 so that only 1000 of each letter are used for training 
	df_training = pd.DataFrame()
	df_training['path'] = ""    ### these prevent some annoying pandas output warnings
	df_training['letter'] = ""
	for iii in range(0,len(datasets.get_train_data())):
		df_training.loc[iii, 'path' ] = datasets.get_train_data()[iii]
		df_training.loc[iii, 'letter' ] = datasets.get_train_labels()[iii] ### converting letter to int
	df_training = df_training.sample(frac=1).reset_index(drop=True) ### shuffle
	print("Writing CSV Files.")
	df_training.to_csv(base_dir+"/processedDatasets/train_data.csv") ### write out csv file 
	letters_binarized = pd.get_dummies(df_training["letter"],dtype=int) ### binarize
	letters_binarized.insert(0, 'path', df_training['path']) ### reinsert the path 
	letters_binarized.to_csv(base_dir+"/processedDatasets/train_data_binarized.csv") ### write out binarized csv
	print("Saved csv files to processedDatasets/")
	print("Now saving labels to pickle file.")
	save_labels(datasets.get_train_labels())
def save_labels(train_labels):
	base_dir = os.path.join('%s'%os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1] ))
	binarized_labels = sklearn.preprocessing.label_binarize(train_labels, classes = np.array("A/B/C/D/E/F/G/H/I/J/K/L/M/N/O/P/Q/R/S/T/U/V/W/X/Y/Z/del/nothing/space".split("/")))
	with open(base_dir+'/processedDatasets/labels_binarized.pickle', 'wb') as handle:
		pickle.dump(binarized_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("Saved binarized labels to processedDatasets/labels_binarized.pickle")
	### test pickle file is working
	#binarized_labels_pkl = open('processedDatasets/labels_binarized.pickle', 'rb')
	# dump information to that file
	#binarized_labels_loaded = pickle.load(binarized_labels_pkl)
	return
def main(args):
	if len(args) != 2:
		print("Incorrect inputs. Use format python3 make_csv.py <int = number of files of each category to load>")
		return
	create_csv(int(args[1]))
	return
if __name__=="__main__":
	main(sys.argv)
