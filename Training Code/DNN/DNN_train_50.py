import csv, time, json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from statistics import mean 

### Read in Preprocessed Data ###

# inputs/features
with open('inputs_50.csv', 'r') as f:
	features = list(csv.reader(f))

features = np.array(features, dtype=float)
print(features.shape)

# outputs
with open('outputs_50.csv', 'r') as f2:
	outputs = list(csv.reader(f2))

outputs = np.array(outputs, dtype=float)
print(outputs.shape)
num_inputs = len(features[0])
outputs = outputs.transpose()
print(outputs.shape)

# convert outputs to one hot encoded
oneHot_outputs = np_utils.to_categorical(outputs)
print(oneHot_outputs.shape)

# define hyperparameters
EPOCHS = 400
BATCH_SIZE = 100
LAYER1_SIZE = 500
LAYER2_SIZE = 50

def model_test_1(EPOCHS, BATCH_SIZE, LAYER1_SIZE, num_inputs=num_inputs):
	# create model
	def my_model():
		model = Sequential()
		model.add(Dense(LAYER1_SIZE, input_dim=num_inputs, activation='relu'))
		model.add(Dense(5, activation='softmax'))
		# compile
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	estimator = KerasClassifier(build_fn=my_model, epochs=EPOCHS, batch_size=BATCH_SIZE)
	# 10-fold cross validation
	cross_val = KFold(n_splits=10, shuffle=True)
	results = cross_val_score(estimator, features, oneHot_outputs, cv=cross_val)
	return results.mean()*100, results.std()*100
	# print("Baseline: %.4f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def model_test_2(EPOCHS, BATCH_SIZE, LAYER1_SIZE, LAYER2_SIZE, num_inputs=num_inputs):
	# create model
	def my_model():
		model = Sequential()
		model.add(Dense(LAYER1_SIZE, input_dim=num_inputs, activation='relu'))
		model.add(Dense(LAYER2_SIZE, activation='relu'))
		model.add(Dense(5, activation='softmax'))
		# compile
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	estimator = KerasClassifier(build_fn=my_model, epochs=EPOCHS, batch_size=BATCH_SIZE)
	# 10-fold cross validation
	cross_val = KFold(n_splits=10, shuffle=True)
	results = cross_val_score(estimator, features, oneHot_outputs, cv=cross_val)
	return results.mean()*100, results.std()*100
	# print("Baseline: %.4f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def model_test_3(EPOCHS, BATCH_SIZE, LAYER1_SIZE, LAYER2_SIZE, LAYER3_SIZE, num_inputs=num_inputs):
	# create model
	def my_model():
		model = Sequential()
		model.add(Dense(LAYER1_SIZE, input_dim=num_inputs, activation='relu'))
		model.add(Dense(LAYER2_SIZE, activation='relu'))
		model.add(Dense(LAYER3_SIZE, activation='relu'))
		model.add(Dense(5, activation='softmax'))
		# compile
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	estimator = KerasClassifier(build_fn=my_model, epochs=EPOCHS, batch_size=BATCH_SIZE)
	# 10-fold cross validation
	cross_val = KFold(n_splits=10, shuffle=True)
	results = cross_val_score(estimator, features, oneHot_outputs, cv=cross_val)
	return results.mean()*100, results.std()*100
	# print("Baseline: %.4f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# Stuff for 1 level
acc = []
sd = []
timings = []
# get average
for x in range(10):
	start_time = time.time()
	accuracy, standard_dev = model_test_1(EPOCHS, BATCH_SIZE, LAYER1_SIZE)
	end_time = time.time()
	acc.append(accuracy)
	sd.append(standard_dev)
	timings.append(end_time - start_time)

accuracy = np.mean(np.array(acc))
standard_dev = np.mean(np.array(sd))
avg_time = np.mean(np.array(timings))

print('DNN (50 ms):')
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
with open('DNN_results_50.json', 'w') as f_out:
	json.dump(results, f_out, indent=4)