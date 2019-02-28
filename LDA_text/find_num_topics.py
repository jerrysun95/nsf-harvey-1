import sys
import argparse
import nltk
from nltk.corpus import stopwords, wordnet

from feed_tweets import *

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--storm", help="what storm name you want to use", default="harvey")
	return parser.parse_args()


def main():
	args = parse_args()	
	print("Testing with {storm}".format(storm=args.storm))

	tokenizer = RegexpTokenizer(r'[a-z0-9\']+')
	p_stemmer = PorterStemmer()


	data_storm = read_data(args.storm)
	_, corpus_storm = parse_text(data_storm, "", tokenizer, en_stop, p_stemmer)
	print("Length of Data: {length}".format(length=len(data_storm)))

	data_noise = read_data("noise")
	_, corpus_noise = parse_text(data_noise, "", tokenizer, en_stop, p_stemmer)
	print("Length of Data: {length}".format(length=len(data_noise)))

	data_sandy = read_data("sandy")
	_, corpus_sandy = parse_text(data_sandy, "", tokenizer, en_stop, p_stemmer)
	print("Length of Data: {length}".format(length=len(data_sandy)))

	h_and_n = []
	s_and_n = []
	h_and_s = []
	for num_topics in range(1, 11):
		lda_storm = run_model(data_storm, args.storm, num_topics)
		lda_noise = run_model(data_noise, "noise", num_topics)
		lda_sandy = run_model(data_sandy, "sandy", num_topics)
		h_and_n.append(compare(lda_storm, corpus_storm, lda_noise, corpus_noise))
		h_and_s.append(compare(lda_storm, corpus_storm, lda_sandy, corpus_sandy))
		s_and_n.append(compare(lda_sandy, corpus_sandy, lda_noise, corpus_noise))
		# h_and_n.append(compare_models(lda_storm, lda_noise))
		# h_and_s.append(compare_models(lda_storm, lda_sandy))
		# s_and_n.append(compare_models(lda_sandy, lda_noise))

	print("harvey and noise")
	for i in range(len(h_and_n)):
		print("k="+str(i+1), h_and_n[i])
	print("sandy and noise")
	for i in range(len(s_and_n)):
		print("k="+str(i+1), s_and_n[i])
	print("harvey and sandy")
	for i in range(len(h_and_s)):
		print("k="+str(i+1), h_and_s[i])



if __name__ == '__main__':
	main()