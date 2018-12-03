import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords, wordnet
from gensim import corpora, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import gensim
from gensim.test.utils import datapath
import json
import time
import glob
import re
import ast

from text import *

nltk.download('stopwords')
en_stop = stopwords.words('english')

def main():
	storm = 'private'
	data = read_data(storm)
	print("Length of Data: {length}".format(length=len(data)))
	lda, dictionary, corpus = run_model(data, storm)
	# visualize(lda, dictionary, corpus)

	# bow = dictionary.doc2bow(["Help", "i", "am", "stuck", "in", "a", "storm"])
	# print(bow)
	# topics = lda.get_document_topics(bow, per_word_topics=True)
	# print("TOPICS-S:", topics)

	# bow = dictionary.doc2bow(["check", "out", "my", "new", "album"])
	# print(bow)
	# topics = lda.get_document_topics(bow, per_word_topics=True)
	# print("TOPICS-N:", topics)



def read_data(storm, includeNoise=False):
	print("Extracting data...")
	data = []

	if storm == 'sandy':
		with open('tweets_sandy/affected_tweets_clean.pkl', 'rb') as f:
			data.extend(pickle.load(f))

		with open('tweets_sandy/destroyed_tweets_clean.pkl', 'rb') as f:
			data.extend(pickle.load(f))

		with open('tweets_sandy/major_tweets_clean.pkl', 'rb') as f:
			data.extend(pickle.load(f))

		with open('tweets_sandy/minor_tweets_clean.pkl', 'rb') as f:
			data.extend(pickle.load(f))

	elif storm == 'harvey':
		files = glob.glob('tweets_harvey/tweets.log*')
		for filename in files:
			print("Extracting file {FILE}".format(FILE=filename))
			f = open(filename, 'r').read()
			indicesStart = [m.start() for m in re.finditer('(?=text.*is_quote_status)', f)]
			indicesEnd = [m.start() for m in re.finditer('(?=is_quote_status)', f)]

			iS = 0
			iE = 0

			while (iS+1 < len(indicesStart) and iE < len(indicesEnd)):
				if indicesStart[iS+1] > indicesEnd[iE]:
					data.append(f[indicesStart[iS]+9:indicesEnd[iE]-5])
					iE += 1
				else:
					iS += 1

			if iE < len(indicesEnd):
				data.append(f[indicesStart[iS]+9:indicesEnd[iE]-5])
	elif storm == 'bonnie':
		with open('bonnie.csv', 'r') as f:
			for line in f:
				data.append(line)
	elif storm == 'Florence' or storm == 'Lane' or storm == 'Michael':
		files = glob.glob('{STORM}/*'.format(STORM=storm))
		for filename in files:
			try:
				with open(filename, 'r') as f:
					print("Extracting file {FILE}".format(FILE=filename))
					headers = next(f).split(',')
					idx = headers.index('title')
					idx2 = headers.index('description')
					for line in f:
						tweets = line.split(',')
						data.append(tweets[idx])
						data.append(tweets[idx2])
			except:
				print("ERROR: Extracting file {FILE} failed".format(FILE=filename))
	elif storm == 'private':
		files = glob.glob('{STORM}/*'.format(STORM=storm))
		for filename in files:
			with open(filename, 'r') as f:
				print("Extracting file {FILE}".format(FILE=filename))
				while True:
					try:
						line = next(f)	
						data.append(line)
					except StopIteration:
						break
					except:
						print("skip")

	if includeNoise:
		files = glob.glob('tweets_noise/tweets_random*')
		for f in files[:len(files)//2]:
			print(f)
			noise = json.load(open(f))
			for j in noise:
				j = j["text"].lower()
				data.append(j)

	return data

def run_model(data, storm):
	tokenizer = RegexpTokenizer(r'[a-z0-9\']+')
	p_stemmer = PorterStemmer()

	dictionary, corpus = parse_text(data, "affected_tweets-1", tokenizer, en_stop, p_stemmer)

	start = time.time() 
	save_file = datapath(storm)
	try:
		lda = gensim.models.ldamodel.LdaModel.load(save_file)
	except FileNotFoundError:
		print("WARNING: Model not found...")
		lda = build_model(dictionary, corpus)
		lda.save(save_file)
	end = time.time()
	print("Time taken to run: {SEC} seconds".format(SEC=end - start))
	

	print("Printing topics...")
	topics = lda.show_topics(formatted=False, log=False)
	res = []
	for t in topics:
		print("NEW TOPIC...")
		for word in t:
			print(word)
		res.append(t)

	return lda, dictionary, corpus

def visualize(ldamodel, dictionary, corpus):
	print("START VISUALIZING...")
	# lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
	# pyLDAvis.display(lda_display)

	K = 5
	topicWordProbMat = ldamodel.print_topics(K)
	print(topicWordProbMat)

	columns = ['1','2','3','4','5']

	df = pd.DataFrame(columns = columns)
	pd.set_option('display.width', 1000)

	# 40 will be resized later to match number of words in DC
	zz = np.zeros(shape=(50, K))

	last_number=0
	DC={}

	for x in range (10):
	  data = pd.DataFrame({columns[0]:"",
	                     columns[1]:"",
	                     columns[2]:"",
	                     columns[3]:"",
	                     columns[4]:"",
	                                                                                       
	                     
	                    },index=[0])
	  df=df.append(data,ignore_index=True)  
	    
	for line in topicWordProbMat:
	    tp, w = line
	    probs=w.split("+")
	    y=0
	    for pr in probs:
	               
	        a=pr.split("*")
	        df.iloc[y,tp] = a[1]
	       
	        if a[1] in DC:
	           zz[DC[a[1]]][tp]=a[0]
	        else:
	           print(last_number, tp)
	           zz[last_number][tp]=a[0]
	           DC[a[1]]=last_number
	           last_number=last_number+1
	        y=y+1

	print (df)
	print (zz)

	
	zz=np.resize(zz,(len(DC.keys()),zz.shape[1]))

	for val, key in enumerate(DC.keys()):
	        plt.text(-2.5, val + 0.5, key,
	                 horizontalalignment='center',
	                 verticalalignment='center'
	                 )

	plt.imshow(zz, cmap='hot', interpolation='nearest')
	plt.show()

def update(data, dictionary):
	print("BEFORE UPDATE...")
	dictionary, new_corpus = parse_text(data, "affected_tweets_clean-2", tokenizer, en_stop, p_stemmer, dictionary=dictionary)
	lda.update(new_corpus)

	bow = dictionary.doc2bow(data)
	print(bow)
	# topics = lda.show_topics(num_topics=5, num_words=15, formatted=False, log=False)
	topics = lda.get_document_topics(bow, per_word_topics=True)

	res = []
	for t in topics:
		print("NEW TOPIC...")
		for word in t:
			print(word)
		res.append(t)

def compare():
	#Train your LDA model    
	lda = LdaModel(national_corpus, num_topics=10)

	# Get the mean of all topic distributions in one corpus
	national_topic_vectors = []
	for newspaper in national_corpus:
		national_topic_vectors.append(lda[newspaper])
	national_average = numpy.average(numpy.array(national_topic_vectors), axis=0)

	# Get the mean of all topic distributions in another corpus
	regional_topic_vectors = []
	for newspaper in regional_corpus:
		regional_topic_vectors.append(lda[newspaper])
	regional_average = numpy.average(numpy.array(regional_topic_vectors), axis=0)

	# Calculate the distance between the distribution of topics in both corpora
	difference_of_distributions = numpy.linalg.norm(national_average - regional_average)

	# Hellinger distance
	# KL divergence
		


if __name__ == '__main__':
	main()