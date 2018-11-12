import pickle
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from gensim import corpora, models
import pandas as pd
import numpy as np
# import pyLDAvis.gensim


import gensim
import json
import time
import glob
import re

from text import *

def main():
	data = read_data()
	lda, dictionary, corpus = run_model(data)
	visualize(lda, dictionary, corpus)

def read_data():
	with open('tweets_sandy/affected_tweets_clean.pkl', 'rb') as f:
		data = pickle.load(f)

	files = glob.glob('tweets_noise/tweets_random*')
	for f in files[:len(files)//2]:
		print(f)
		noise = json.load(open(f))
		for j in noise:
			j = j["text"].lower()
			data.append(j)

	return data

def run_model(data):
	tokenizer = RegexpTokenizer(r'[a-z0-9\']+')

	# create English stop words list
	en_stop = get_stop_words('en')

	# Create p_stemmer of class PorterStemmer
	p_stemmer = PorterStemmer()

	dictionary, corpus = parse_text(data, "affected_tweets-1", tokenizer, en_stop, p_stemmer)
	start = time.time()
	lda = build_model(dictionary, corpus)
	end = time.time()
	print(end - start)
	

	print("Printing topics")
	topics = lda.show_topics(num_topics=5, num_words=15, formatted=False, log=False)
	res = []
	for t in topics:
		print("NEW TOPIC")
		for word in t:
			print(word)
		res.append(t)

	# lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
	# pyLDAvis.display(lda_display)
	return lda, dictionary, corpus

def visualize(ldamodel, dictionary, corpus):
	print("START VISUALIZING")
	K = 5
	topicWordProbMat = ldamodel.print_topics(K)
	print(topicWordProbMat)

	columns = ['1','2','3','4','5']

	df = pd.DataFrame(columns = columns)
	pd.set_option('display.width', 1000)

	# 40 will be resized later to match number of words in DC
	zz = np.zeros(shape=(40,K))

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

def update(data):
	print("BEFORE UPDATE")
	dictionary, new_corpus = parse_text(data[1000:2000], "affected_tweets_clean-2", tokenizer, en_stop, p_stemmer, dictionary=dictionary)
	lda.update(new_corpus)

	topics = lda.show_topics(num_topics=5, num_words=15, formatted=False, log=False)
	res = []
	for t in topics:
		print("NEW TOPIC")
		for word in t:
			print(word)
		res.append(t)


	


if __name__ == '__main__':
	main()