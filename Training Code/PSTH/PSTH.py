import json, math, sys, time
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

with open('hw4.json') as f:
	data = json.load(f)

pretime = 0
posttime = 0.2

# creates the template for the event that the test trial is from (the "true" event)
def perNeuron_true(neuron_matrix, index):
	temp_rrm = []
	# splits rrm into template and test trial
	for i, trial in enumerate(neuron_matrix):
		if i != index:
			temp_rrm.append(trial)
		else:
			test_trial = trial

	template_neuron = [(sum(i)/(len(temp_rrm))) for i in zip(*temp_rrm)]

	return template_neuron, test_trial

### Outputs a true label and the predicted label for each inputted trial ###
def true_pred(bin_size, event, test_trial_index, template_dict, rrm, pre_time=pretime, post_time=posttime):
	# Template for "true" event
	template = []
	# Template for "false" event (from template_dict)
	template_false = []
	# Test trial
	testing_trial = []

	# template for true event
	temp_template = []
	temp_trial = []

	template_start_time = time.time()
	for n in rrm:
		temp_template, temp_trial = perNeuron_true(n, test_trial_index)
		template.append(temp_template)
		testing_trial.append(temp_trial)
	template_end_time = time.time()

	# run through all neurons (each list in template) and calculate "inner part" of Euclidian distance
	# add each to a sum
	sum_neuron = 0.0
	sum_distance = 0.0
	temp = 0.0
	for i, neuron in enumerate(template):
		sum_neuron = 0.0
		for j, n in enumerate(neuron):
			temp = (n - testing_trial[i][j]) ** 2
			sum_neuron += temp
		sum_distance += sum_neuron
	true_distance = math.sqrt(sum_distance)
	label = event

	# compare to all other events
	prev_distance = true_distance


	for e in data['events']:
		template_false = template_dict[e]

		# skip over true event
		if e == event:
			continue

		# Calculate Eulidian distance from false event
		sum_neuron_false = 0.0
		sum_distance_false = 0.0
		temp_false = 0.0
		for i, neuron_false in enumerate(template_false):
			sum_neuron_false = 0.0
			for j, n_false in enumerate(neuron_false):
				temp_false = (n_false - testing_trial[i][j]) ** 2
				sum_neuron_false += temp_false
			sum_distance_false += sum_neuron_false

		false_distance = math.sqrt(sum_distance_false)
		sum_distance_false = 0.0

		# Compare distances to each other to find the smallest
		if false_distance < prev_distance:
			prev_distance = false_distance
			label = e
		false_distance = 0.0
	false_distance = 0.0
	prev_distance = 0.0

	# returns predicted label ("label"), true label ("event"), and time to make the true template ("template_time")
	template_time = template_end_time - template_start_time
	return label, event, template_time


### Implement all for each bin size ###

# Bin size: 5 ms
with open('template_dict_5.json') as f1:
	template_dict_5ms = json.load(f1)
print('5 ms loaded.')

with open('PSTH_RRM.json') as f_rrm1:
	rrm_5 = json.load(f_rrm1)

# go through every event getting to be the "true label" and each bin size
bin_size = 0.005
predicted_events_5ms = []
true_events_5ms = []
temp_full_time = []
full_time = []
time_without_template = []
temp_time_without_template = []
performance_5ms = []
# create predicted and true label lists via classification and cross validation
for runthrough1 in range(10):
	for e in data['events']:
		for i, event_number in enumerate(data['events'][e]):
			rrm_event = rrm_5[e]
			start_time = time.time()
			pred_label, true_label, temp_time = true_pred(bin_size, e, i, template_dict_5ms, rrm_event)
			end_time = time.time()
			predicted_events_5ms.append(pred_label)
			true_events_5ms.append(true_label)
			temp_full_time.append(end_time - start_time)
			temp_time_without_template.append(end_time - (start_time + temp_time))
		print(e + ' completed for 5 ms')
	performance_5ms.append(accuracy_score(true_events_5ms, predicted_events_5ms))
	full_time.append(np.sum(np.array(temp_full_time)))
	time_without_template.append(np.sum(np.array(temp_time_without_template)))
	print(str(runthrough1+1) + ' for 5 ms done!')

# find the mean performance, mean time, and mean of the time without the creation of the template
accuracy_5ms = np.mean(np.array(performance_5ms))
full_time_5ms = np.mean(np.array(full_time))
time_without_template_5ms = np.mean(np.array(time_without_template))

# Save Results
results = {}
results['accuracy'] = performance_5ms
results['full_timings'] = full_time
results['time_without_template'] = time_without_template
results['average accuracy'] = accuracy_5ms
results['average full time'] = full_time_5ms
results['average time without template'] = time_without_template_5ms

# write results to json file
with open('PSTH_results.json', 'w') as f_out:
	json.dump(results, f_out, indent=4)

print('5 ms: completed')

# Bin size: 50 ms
with open('template_dict_50.json') as f2:
	template_dict_50ms = json.load(f2)

print('50 ms loaded.')

with open('PSTH_RRM_50.json') as f_rrm2:
	rrm_50 = json.load(f_rrm2)

# go through every event getting to be the "true label" and each bin size
bin_size = 0.05
predicted_events_50ms = []
true_events_50ms = []
temp_full_time = []
full_time = []
time_without_template = []
temp_time_without_template = []
performance_50ms = []
# create predicted and true label lists via classification and cross validation
for runthrough2 in range(10):
	for e in data['events']:
		for i, event_number in enumerate(data['events'][e]):
			rrm_event = rrm_50[e]
			start_time = time.time()
			pred_label, true_label, temp_time = true_pred(bin_size, e, i, template_dict_50ms, rrm_event)
			end_time = time.time()
			predicted_events_50ms.append(pred_label)
			true_events_50ms.append(true_label)
			temp_full_time.append(end_time - start_time)
			temp_time_without_template.append(end_time - (start_time + temp_time))
		print('...........' + e + ' completed for 50 ms')
	performance_50ms.append(accuracy_score(true_events_50ms, predicted_events_50ms))
	full_time.append(np.sum(np.array(temp_full_time)))
	time_without_template.append(np.sum(np.array(temp_time_without_template)))
	print(str(runthrough2+1) + ' for 50 ms done!')

# find the mean performance, mean time, and mean of the time without the creation of the template
accuracy_50ms = np.mean(np.array(performance_50ms))
full_time_50ms = np.mean(np.array(full_time))
time_without_template_50ms = np.mean(np.array(time_without_template))

# Save Results
results = {}
results['accuracy'] = performance_50ms
results['full_timings'] = full_time
results['time_without_template'] = time_without_template
results['average accuracy'] = accuracy_50ms
results['average full time'] = full_time_50ms
results['average time without template'] = time_without_template_50ms

# write results to json file
with open('PSTH_results_50.json', 'w') as f_out2:
	json.dump(results, f_out2, indent=4)

print('50 ms: completed')

# Bin size: 200 ms
with open('template_dict_200.json') as f3:
	template_dict_200ms = json.load(f3)

print('200 ms loaded.')

with open('PSTH_RRM_200.json') as f_rrm3:
	rrm_200 = json.load(f_rrm3)

bin_size = 0.2
predicted_events_200ms = []
true_events_200ms = []
temp_full_time = []
full_time = []
time_without_template = []
temp_time_without_template = []
performance_200ms = []
# create predicted and true label lists via classification and cross validation
for runthrough3 in range(10):
	for e in data['events']:
		for i, event_number in enumerate(data['events'][e]):
			rrm_event = rrm_200[e]
			start_time = time.time()
			pred_label, true_label, temp_time = true_pred(bin_size, e, i, template_dict_200ms, rrm_event)
			end_time = time.time()
			predicted_events_200ms.append(pred_label)
			true_events_200ms.append(true_label)
			temp_full_time.append(end_time - start_time)
			temp_time_without_template.append(end_time - (start_time + temp_time))
		print(e + ' completed for 200 ms')
	performance_200ms.append(accuracy_score(true_events_200ms, predicted_events_200ms))
	full_time.append(np.sum(np.array(temp_full_time)))
	time_without_template.append(np.sum(np.array(temp_time_without_template)))
	print(str(runthrough3+1) + ' for 200 ms done!')

# find the mean performance, mean time, and mean of the time without the creation of the template
accuracy_200ms = np.mean(np.array(performance_200ms))
full_time_200ms = np.mean(np.array(full_time))
time_without_template_200ms = np.mean(np.array(time_without_template))

# Save Results
results = {}
results['accuracy'] = performance_200ms
results['full_timings'] = full_time
results['time_without_template'] = time_without_template
results['average accuracy'] = accuracy_200ms
results['average full time'] = full_time_200ms
results['average time without template'] = time_without_template_200ms

# write results to json file
with open('PSTH_results_200.json', 'w') as f_out3:
	json.dump(results, f_out3, indent=4)

print('200 ms: completed')