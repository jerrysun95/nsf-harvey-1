import json, random
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier as nn
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.naive_bayes import BernoulliNB as bnb
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import matplotlib.pyplot as plt

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
				fmap[d['name']] = 0

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
		rec = [0] * len(fmap)

		for a in d['attributes']:
			if a in fmap:
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
def _svm(t, min_freq):
	# svc = svm.SVC()
	# parameters = {'C':[.9 + 0.01*x for x in range(1,5)], 
	# 			  'kernel':['linear', 'poly', 'rbf'], 
	# 			  'shrinking':[True, False]}
	# clf = GridSearchCV(svc, parameters, cv=5)
	# clf.fit(records, labels)
	# save_classifier(clf.best_estimator_, t, 'svm', min_freq)
	# return ('svc', clf.best_estimator_)
	clf = load_classifier(t, 'svm', min_freq)
	return ('svc', clf)

# Creates predictive model using naive bayes algorithm and train_data
# Returns the accuracy of the model using test_data
def _gnb(t, min_freq):
	# clf = gnb().fit(records, labels)
	# save_classifier(clf, t, 'gnb', min_freq)
	# return ('gnb', clf)
	clf = load_classifier(t, 'gnb', min_freq)
	return ('gnb', clf)

def _mnb(t, min_freq):
	# clf = mnb().fit(records, labels)
	# save_classifier(clf, t, 'mnb', min_freq)
	# return ('mnb', clf)
	clf = load_classifier(t, 'mnb', min_freq)
	return ('mnb', clf)

def _bnb(t, min_freq):
	# clf = bnb().fit(records, labels)
	# save_classifier(clf, t, 'bnb', min_freq)
	# return ('bnb', clf)
	clf = load_classifier(t, 'bnb', min_freq)
	return ('bnb', clf)

# Creates predictive model using nearest neighbors algorithm and train_data
# Returns the accuracy of the model using test_data
def _knn(t, min_freq):
	# nbrs = nn()
	# parameters = {'n_neighbors':[1], 
	# 			  'weights':['uniform', 'distance'], 
	# 			  'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
	# clf = GridSearchCV(nbrs, parameters, cv=5)
	# clf.fit(records, labels)
	# save_classifier(clf.best_estimator_, t, 'knn', min_freq)
	# return ('knn', clf.best_estimator_)
	clf = load_classifier(t, 'knn', min_freq)
	return ('knn', clf)

def _dt(t, min_freq):
	# clf = tree.DecisionTreeClassifier().fit(records, labels)
	# save_classifier(clf, t, 'dt', min_freq)
	# return ('dt', clf)
	clf = load_classifier(t, 'dt', min_freq)
	return ('dt', clf)

def _sgd(t, min_freq):
	# s = sgd()
	# parameters = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
	# 			  'penalty': ['l2', 'l1', 'none', 'elasticnet'],
	# 			  'alpha': [.00001]}
	# clf = GridSearchCV(s, parameters, cv=5)
	# clf.fit(records, labels)
	# save_classifier(clf.best_estimator_, t, 'sgd', min_freq)
	# return ('sgd', clf.best_estimator_)
	clf = load_classifier(t, 'sgd', min_freq)
	return ('sgd', clf)

def _mlp(t, min_freq):
	# clf = mlp().fit(records, labels)
	# save_classifier(clf, t, 'mlp', min_freq)
	# return ('mlp', clf)
	clf = load_classifier(t, 'mlp', min_freq)
	return ('mlp', clf)

def save_classifier(clf, typ, name, min_freq):
	s = str(min_freq).replace('.', '-')
	joblib.dump(clf, 'classifiers/' + typ + '_' + name + '_' + s + '.pkl')

def load_classifier(typ, name, min_freq):
	s = str(min_freq).replace('.', '-')
	return joblib.load('classifiers/' + typ + '_' + name + '_' + s + '.pkl')

def voting_classifier(freq, data):
	fmap = {}
	aggregate_freq(freq)

	global records, labels
	records = []
	labels = []
	label = 0
	for d in data:
		iter_results(d, label)
		label += 1

	z = zip(records, labels)
	random.shuffle(z)

	records = [x[0] for x in z]
	labels = [x[1] for x in z]

	estimators = []
	print 'svm'
	estimators.append(_svm())
	print 'gnb'
	estimators.append(_gnb())
	print 'mnb'
	estimators.append(_mnb())
	print 'bnb'
	estimators.append(_bnb())
	print 'knn'
	estimators.append(_knn())
	print 'dt'
	estimators.append(_dt())
	print 'sgd'
	estimators.append(_sgd())
	print 'mlp'
	estimators.append(_mlp())

	eclf = VotingClassifier(estimators=estimators, voting='hard')
	scores = cross_val_score(eclf, records, labels, cv=5)
	print(scores)
	print(scores.mean())

	return eclf, scores.mean()

def stacking_classifier(freq, data, typ, min_freq):
	fmap = {}
	aggregate_freq(freq)

	global records, labels
	records = []
	labels = []
	label = 0
	for d in data:
		iter_results(d, label)
		label += 1

	z = zip(records, labels)
	random.shuffle(z)

	records = [x[0] for x in z]
	labels = [x[1] for x in z]

	pred = []
	accs = []
	classifiers = [_svm, _gnb, _mnb, _bnb, _knn, _dt, _sgd, _mlp]
	for c in classifiers:
		clf = c(typ, min_freq)[1]
		pred.append(clf.predict(records))
		accs.append(cross_val_score(clf, records, labels, cv=5).mean())
	
	stacked_records = zip(*pred)

	# svc = svm.SVC()
	# parameters = {'C':[.9 + 0.01*x for x in range(1,5)], 
	# 			  'kernel':['linear'], 
	# 			  'shrinking':[True, False]}
	# clf = GridSearchCV(svc, parameters, cv=5)
	# clf.fit(stacked_records, labels)
	# save_classifier(clf.best_estimator_, typ, 'stack', min_freq)
	
	clf = load_classifier(typ, 'stack', min_freq)
	
	scores = cross_val_score(clf, stacked_records, labels, cv=5)
	print(scores)
	print(str(scores.mean()) + '\n')

	# clf.fit(stacked_records, labels)

	# x_train, x_test, y_train, y_test = train_test_split(stacked_records, labels)
	# y_pred = clf.fit(x_train, y_train).predict(x_test)	
	# print(confusion_matrix(y_test, y_pred))

	# pca = PCA(n_components=2)
	# data = pca.fit_transform(stacked_records)
	# plt.scatter([x[0] for x in data], [y[1] for y in data], c=labels)
	# plt.show()
	
	return clf, scores.mean(), accs
