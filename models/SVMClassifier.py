#=========================================================#
#                     author:                             #                           
#                     date:                               #
#                     dateset:                            #
#                     model:                              #
#=========================================================#

#packages
import numpy as np

#SVM
class SVM(object):
	'''SVM class
	'''
	def __init__(self, input_size, output_size, opt):
		self.num_classes = output_size

		self.W = opt.SVM_standard * np.random.randn(input_size, output_size)

	def loss(self, X, y, opt):   #opt: reg
		output = X.dot(self.W)
		correct_classes = output[range(opt.batch_size), list(y)].reshape(-1, 1)

		margins = np.maximum(0, scores - correct_classes + 1)   #svm loss
		margins[range(num_train), list(y)] = 0

		loss = np.sum(margins) / num_train + 0.5 * opt.SVM_regulization * np.sum(self.W * self.W)

		coeff_mat = np.zeros((opt.batch_size, self.num_classes))
		coeff_mat[margins > 0] = 1
		coeff_mat[range(opt.batch_size), list(y)] = 0
		coeff_mat[range(opt.batch_size), list(y)] = -np.sum(coeff_mat, axis=1)

		grad = (X.T).dot(coeff_mat)
		grad = grad / opt.batch_size + opt.SVM_regulization * self.W

		return loss, grad

	def train(self, X, y, X_val, y_val, opt):
		#opt: leanring_rate, reg, epoch, batch_size
		self.num_train, _ = X.shape

		loss_log = []
		train_acc_log = []
		val_acc_log = []

		for e in range(opt.epoch):
			batch_idx = np.random.choice(self.num_train, opt.batch_size, replace=True)

			X_batch = X[batch_idx]
			y_batch = y[batch_idx]

			loss, grad = self.loss(X_batch, y_batch, opt)
			loss_log.append(loss)

			self.W -= learning_rate * grad

			if e % 100 == 0:
				train_acc = (self.predict(X_batch) == y_batch).mean()
				val_acc = (self.predict(X_val) == y_val).mean()

				print('epoch {} / {}: loss {}, acc {}, val {}'.format(e, opt.epoch, loss, train_acc, val_acc))

				train_acc_log.append(train_acc)
				val_acc_log.append(val_acc)

		return {'loss_log': loss_log, 'train_acc_log': train_acc_log, 'val_acc_log': val_acc_log}

	def predict(self, X):
		output = X.dot(self.W)
		pred = np.argmax(output, axis=1)

		return pred
