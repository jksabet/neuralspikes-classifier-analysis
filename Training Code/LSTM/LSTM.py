# LSTM
import json, time
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import numpy as np

# Get list of neurons
with open('hw4.json') as f:
	data = json.load(f)

neuron_list = list(data['neurons'])
print(neuron_list)

# load a single file as an array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames):
	loaded = list()
	for name in filenames:
		data = load_file(name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# load a dataset group, such as train or test
def load_dataset_group():
	# load all 9 files as a single array
	filenames = list()
	# save each neuron filename
	for n in neuron_list:
		f = n + '.txt'
		filenames.append(f)
	# load input data
	X = load_group(filenames)
	# load class output
	y = load_file('y_LSTM.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all
	X, y = load_dataset_group()
	print(X.shape, y.shape)

	# zero-offset class values
	y = y - 1

	# one hot encode y
	y = to_categorical(y)
	print(X.shape, y.shape)
	return X, y

# fit and evaluate a model
def run_model(X, y):
	EPOCHS, BATCH_SIZE = 400, 100
	n_timesteps, n_features, n_outputs = X.shape[1], X.shape[2], y.shape[1]

	def my_model():
		model = Sequential()
		model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
		model.add(Dropout(0.5))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(n_outputs, activation='softmax'))
		# compile
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	estimator = KerasClassifier(build_fn=my_model, epochs=EPOCHS, batch_size=BATCH_SIZE)
	# 10-fold cross validation
	cross_val = KFold(n_splits=10, shuffle=True)
	results = cross_val_score(estimator, X, y, cv=cross_val)
	return results.mean()*100, results.std()*100

# Average of 10 runs: accuracy, timing, and standard deviation
acc = []
sd = []
timings = []
x, y = load_dataset()

for i in range(10):
	start_time = time.time()
	accuracy, standard_dev = run_model(x, y)
	end_time = time.time()
	timings.append(end_time - start_time)
	acc.append(accuracy)
	sd.append(standard_dev)

accuracy = np.mean(np.array(acc))
standard_dev = np.mean(np.array(sd))
avg_time = np.mean(np.array(timings))


print("Baseline: %.4f%% (%.2f%%)" % (accuracy, standard_dev))
print('Time Taken: %.4f seconds' % (avg_time))

# Save Results
results = {}
results['accuracy'] = acc
results['standard_dev'] = sd
results['timings'] = timings
results['average accuracy'] = accuracy
results['average standard_dev'] = standard_dev
results['average timing'] = avg_time

# write results to json file
with open('LSTM_results.json', 'w') as f_out:
	json.dump(results, f_out, indent=4)
