import json
from sklearn import svm
from sklearn.neighbors import NearestNeighbors as nn
from sklearn.naive_bayes import GaussianNB as gnb

# Maps frequent attribute to an index
fmap = {}

# Stores all records and labels of data
records = []
labels = []

# Partitions of records and labels used for model training and evaluation
test_data = []
test_labels = []
train_data = []
train_labels = []

# Respondent type labels
OR = 0
VR = 1
R = 2

# Aggregates all frequent attributes in each file into one set
def aggregate_freq(files):
	for file in files:
		with open('output/' + file + '_attributes.json') as f:
			data = json.loads(f.read())
			for d in data:
				fmap[d] = 0

	# Map attribute to an index for quicker compilation of records later on
	index = 0
	for f in fmap:
		fmap[f] = index
		index += 1

# Runs through results file and adds each item into records and labels
def iter_results(t, label):
	with open('output/' + t + '_results.json') as f:
		data = json.loads(f.read())

	for d in data:
		name = d['name']
		rec = [0] * len(freq)

		for a in d['attributes']:
			i = fmap[a]
			rec[i] = 1

		records.append(rec)
		labels.append(label)

# Partitions records and labels into test and train sets
# Splits on 80/20 train/test
# Must be run before any results are given from machine learning algorithms
# Eventually want to implement some sort of bagging/random sampling partition
# Eventually want to allow for partition to be called multiple times to get different subsets
def partition_data():
	i = 0
	for r in range(len(records)):
		if i % 5 == 0:
			test_data.append(records[i])
			test_labels.append(labels[i])
		else:
			train_data.append(records[i])
			train_labels.append(labels[i])
		i += 1

# Creates predictive model using svm algorithm and train_data
# Returns the accuracy of the model using test_data
def svm(gamma=0.001, C=100.):
	clf = svm.SVC(gamma=gamma, C=C)
	clf.fit(train_data, train_labels)
	
	correct = 0
	for [p, a] in zip(clf.predict(test_data), test_labels):
		if a == p:
			correct += 1
	accuracy = float(correct) / len(test_labels)
	return accuracy

# Creates predictive model using naive bayes algorithm and train_data
# Returns the accuracy of the model using test_data
def naive_bayes()
	g = gnb()
	g.fit(train_data, train_labels)

	correct = 0
	for [p, a] in zip(g.predict(test_data), test_labels):
		if a == p:
			correct += 1
	accuracy = float(correct) / len(test_labels)
	return accuracy

# Creates predictive model using nearest neighbors algorithm and train_data
# Returns the accuracy of the model using test_data
def nearest_neighbors(n_neighbors, algorithm='auto'):
	nbrs = nn(n_neighbors=n_neighbors, algorithm=algorithm).fit(train_data)

	correct = 0
	for [p, a] in zip(nbrs.predict(test_data), test_labels):
		if a == p:
			correct += 1
	accuracy = float(correct) / len(test_labels)
	return accuracy
