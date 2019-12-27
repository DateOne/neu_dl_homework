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
#import pandas as pd

import time

from models.NNClassifier import NeuralNetwork

#data loader
def dataloader(opt):
	'''statlogvehiclesilhouettes dataloader
	read data from txt in file root as np.array, numerize labels, split datasets
	'''
	x = []
	y = []

	for filename in os.listdir(opt.root):
		if 'dat' in filename:
			with open(os.path.join(opt.root, filename), 'r') as f:
				x_file = []
				y_file = []

				for line in f:
					x_line = line.split()[:-1]
					y_line = line.split()[-1]

					x_line = list(map(int, x_line))

					x_file.append(x_line)
					y_file.append(y_line)

				x = x + x_file
				y = y + y_file

	classes = {'opel':0, 'saab':1, 'bus':2, 'van':3}
	y_label = []

	for item in y:
		y_label.append(classes[item])

	x = np.array(x)
	y = np.array(y_label)

	#print('x shape is {}, y shape is {}'.format(x.shape, y.shape))   #check size

	training_data = x[: opt.trainingset_size]
	validation_data = x[opt.trainingset_size :]

	training_label = y[: opt.trainingset_size]
	validation_label = y[opt.trainingset_size :]

	return training_data, training_label, validation_data, validation_label

if __name__ == '__main__':
	parser = argparse.ArgumentParser('neural network')

	parser.add_argument(
		'-e', '--epoch', type=int,
		help='number of epoch',
		default=100000)
	parser.add_argument(
		'-b', '--batch_size', type=int,
		help='batch size',
		default=20)
	parser.add_argument(
		'-lr', '--learning_rate', type=float,
		help='learning rate',
		default=8e-5)
	parser.add_argument(
		'-lr_dk', '--learning_rate_decay', type=float,
		help='learning rate decay',
		default=0.999995)
	parser.add_argument(
		'-h_s', '--hidden_size', type=int,
		help='neural network hidden layer size',
		default=12)
	parser.add_argument(
		'-std', '--standard', type=float,
		help='initialization standard deviation',
		default=1e-4)
	parser.add_argument(
		'-reg', '--regulization', type=float,
		help='regulization term',
		default=1e-5)
	parser.add_argument(
		'-r', '--root',
		help='data root',
		default="datasets/StatlogVehicleSilhouettes")
	parser.add_argument(
		'-tr_s', '--trainingset_size', type=int,
		help='the size of the training size',
		default=680)

	args = parser.parse_args()

	X, y, X_val, y_val = dataloader(args)

	input_size = 18
	num_classes = 4

	model = NeuralNetwork(input_size, num_classes, args)

	log = model.train(X, y, X_val, y_val, args)

	plt.subplot(2, 1, 1)
	plt.plot(log['loss_log'])
	plt.title('Loss')
	plt.xlabel('batch')
	plt.ylabel('loss')

	plt.subplot(2, 1, 2)
	plt.plot(log['train_acc_log'], label='train')
	plt.plot(log['val_acc_log'], label='val')
	plt.title('accuracy')
	plt.xlabel('epoch')
	plt.ylabel('acc')
	plt.show()
