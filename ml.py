import json, random
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier as nn
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.naive_bayes import BernoulliNB as bnb
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import SGDClassifier as sgd
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

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
def _svm():
	svc = svm.SVC()
	parameters = {'C':[.9 + 0.01*x for x in range(1,5)], 
				  'kernel':['linear', 'poly', 'rbf'], 
				  'shrinking':[True, False]}
	clf = GridSearchCV(svc, parameters, cv=5)
	clf.fit(records, labels)
	return ('svc', clf.best_estimator_)

# Creates predictive model using naive bayes algorithm and train_data
# Returns the accuracy of the model using test_data
def _gnb():
	return ('gnb', gnb())

def _mnb():
	return ('mnb', mnb())

def _bnb():
	return ('bnb', bnb())

# Creates predictive model using nearest neighbors algorithm and train_data
# Returns the accuracy of the model using test_data
def _knn():
	nbrs = nn()
	parameters = {'n_neighbors':[1], 
				  'weights':['uniform', 'distance'], 
				  'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
	clf = GridSearchCV(nbrs, parameters, cv=5)
	clf.fit(records, labels)
	return ('knn', clf.best_estimator_)

def _dt():
	return ('dt', tree.DecisionTreeClassifier())

def _sgd():
	s = sgd()
	parameters = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
				  'penalty': ['l2', 'l1', 'none', 'elasticnet'],
				  'alpha': [.00001]}
	clf = GridSearchCV(s, parameters, cv=5)
	clf.fit(records, labels)
	return ('sgd', clf.best_estimator_)

def _mlp():
	return ('mlp', mlp())

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

# aggregate_freq(['vr', 'r', 'or'])
# iter_results('vr', VR)
# iter_results('r',  R)
# iter_results('or', OR)

	# aggregate_freq(['overall'])
	# iter_results('overall', 0)
	# iter_results('noise', 1)

# estimators = []
# Naive Bayes
# Multinomial Bayes
# m = mnb()
# bagging = BaggingClassifier(m, max_samples=0.4, max_features=0.4)
# scores = cross_val_score(bagging, records, labels, cv=5)
# estimators.append(('mnb', m))
# scores = cross_val_score(g, records, labels, cv=5)

# print('random forests')
# rf = RandomForestClassifier(n_estimators=100)
# scores = cross_val_score(rf, records, labels, cv=5)
# print(scores)
# print('average accuracy: ' + str(scores.mean()) + '\n')

# print('extremely random trees')
# erf = ExtraTreesClassifier(n_estimators=100)
# scores = cross_val_score(erf, records, labels, cv=5)
# print(scores)
# print('average accuracy: ' + str(scores.mean()) + '\n')

# print('adaboost')
# abc = AdaBoostClassifier(n_estimators=50)
# scores = cross_val_score(abc, records, labels, cv=5)
# print(scores)
# print('average accuracy: ' + str(scores.mean()) + '\n')

# # # Gaussian Bayes
# g = gnb()	
# estimators.append(('gnb', g))
# # scores = cross_val_score(g, records, labels, cv=5)
# # print(scores)
# # print('average accuracy: ' + str(scores.mean()) + '\n')

# # # Bernoulli Bayes
# b = bnb()
# estimators.append(('bnb', b))
# # scores = cross_val_score(g, records, labels, cv=5)
# # print(scores)
# # print('average accuracy: ' + str(scores.mean()) + '\n')

# # Nearest Neighbor
# nbrs = nn()
# parameters = {'n_neighbors':[x for x in range(1,10)], 
# 			  'weights':['uniform', 'distance'], 
# 			  'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'], 
# 			  'leaf_size':[x for x in range(25,36)], 
# 			  'p':[1,2]}
# parameters = {'n_neighbors':[x for x in range(1,20)]}
# clf = GridSearchCV(nbrs, parameters, cv=5)
# clf.fit(records, labels)
# estimators.append(('knn', clf.best_estimator_))

# # SVM
# svc = svm.SVC()
# parameters = {'C':[1.1 - 0.01*x for x in range(1,20)], 
# 			  'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 
# 			  'shrinking':[True, False],
# 			  'probability':[True, False]}
# clf = GridSearchCV(svc, parameters, cv=5)
# clf.fit(records, labels)
# # print(clf.cv_results_['mean_test_score'])
# estimators.append(('svc', clf.best_estimator_))
# # print(my_svm())
# # print(clf.best_score_)
# # print(clf.best_params_)
# # print(clf.best_estimator_)

# # Decision Trees
# clf = tree.DecisionTreeClassifier()
# estimators.append(('dt', clf))
# # scores = cross_val_score(clf, records, labels, cv=5)
# # print(scores)
# # print(scores.mean())

# # SGD
# clf = sgd()
# # scores = cross_val_score(clf, records, labels, cv=5)
# # print(scores)
# # print(scores.mean())
# parameters = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
# 			  'penalty': ['l2', 'l1', 'none', 'elasticnet'],
# 			  'alpha': [.00001 * x for x in 1,16]}
# c = GridSearchCV(clf, parameters, cv=5)
# c.fit(records, labels)
# estimators.append(('sgd', c))
# # print(c.cv_results_['mean_test_score'])
# # print(c.best_score_)
# # print(c.best_params_)

# # Multilayer Perceptron
# clf = mlp()
# # scores = cross_val_score(clf, records, labels, cv=5)
# # print(scores)
# # print(scores.mean())
# # parameters = {'hidden_layer_sizes': (80,),
# #  			  'activation': ['identity', 'logistic', 'tanh', 'relu'],
# #  			  'solver': ['lbfgs', 'sgd', 'adam'],
# #  			  'alpha': [.00001 * x for x in 1,16],
# #  			  'learning_rate': ['constant', 'invscaling', 'adaptive']}
# # c = GridSearchCV(clf, parameters, cv=5)
# # c.fit(records, labels)
# # print(c.cv_results_['mean_test_score'])
# # print(c.best_score_)
# # print(c.best_params_)
# # g.fit(records, labels)
# estimators.append(('mlp', clf))

# # Voting Classifier
# # clf1 = mnb()
# # clf2 = gnb()
# # clf3 = tree.DecisionTreeClassifier()
# # clf4 = nn()
# # clf5 = svm.SVC()
# # clf6 = sgd()
# # clf7 = mlp()

# # eclf = VotingClassifier(estimators=[('mnb', clf1), ('gnb', clf2), ('dt', clf3), ('knn', clf4), ('svc', clf5), ('sgd', clf6), ('mlp', clf7)],
# # 						voting='hard')
# eclf = VotingClassifier(estimators = estimators, voting='hard')
# # for clf in [clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf]:
# # 	scores = cross_val_score(clf, records, labels, cv=5)
# # 	print(scores)
# # 	print(scores.mean())

# scores = cross_val_score(eclf, records, labels, cv=5)
# print(scores)
# print(scores.mean())