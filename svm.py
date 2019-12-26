#=========================================================#
#                     author:                             #                           
#                     date:                               #
#                     dateset:                            #
#                     model:                              #
#=========================================================#

#packages
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time

#argument parser
parser = argparse.ArgumentParser('StatlogVehicleSilhouettes with SVM')

parser.add_argument(
	'-r', '--dataset_root',
	help='dataset directory',
	default='../datasets/StatlogVehicleSilhouettes')

args = parser.parse_args()

#set up dataset
def SVSdataset(opt):
	x = []
	y = []

	for filename in os.listdir(opt.dataset_root):
		if 'dat' in filename:
			with open(os.path.join(opt.dataset_root, filename), 'r') as f:
				x_file = [] 
				y_file = []

				for line in f:
					x_line = line.split()[: -1]
					y_line = line.split()[-1]
					x_line = list(map(int, x_line))
					y_line = list(map(int, y_line))

					x_file.append(x_line)
					y_file.append(y_line)

				x = x + x_file
				y = y + y_file

	return x, y

