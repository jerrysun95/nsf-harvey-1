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
	try:
		with open("storm_extracts/dict_storm", 'rb') as f:
			dict_storm = pickle.load(f)
		with open("storm_extracts/counts_storm", 'rb') as f:
			counts_storm = pickle.load(f)
	except:
		dict_storm, _, counts_storm = parse_text(data_storm, args.storm, tokenizer, en_stop, p_stemmer)
		print("Length of Data: {length}".format(length=len(data_storm)))
		with open("storm_extracts/dict_storm", "wb") as fp:   #Pickling
			pickle.dump(dict_storm, fp)
		with open("storm_extracts/counts_storm", "wb") as fp:   #Pickling
			pickle.dump(counts_storm, fp)

	data_noise = read_data("noise")
	try:
		with open("storm_extracts/dict_noise", 'rb') as f:
			dict_noise = pickle.load(f)
		with open("storm_extracts/counts_noise", 'rb') as f:
			counts_noise = pickle.load(f)
	except:
		dict_noise, _, counts_noise = parse_text(data_noise, "noise", tokenizer, en_stop, p_stemmer)
		print("Length of Data: {length}".format(length=len(data_noise)))
		with open("storm_extracts/dict_noise", "wb") as fp:   #Pickling
			pickle.dump(dict_noise, fp)
		with open("storm_extracts/counts_noise", "wb") as fp:   #Pickling
			pickle.dump(counts_noise, fp)

	data_sandy = read_data("sandy")	
	try:
		with open("storm_extracts/dict_sandy", 'rb') as f:
			dict_sandy = pickle.load(f)
		with open("storm_extracts/counts_sandy", 'rb') as f:
			counts_sandy = pickle.load(f)
	except:
		dict_sandy, _, counts_sandy = parse_text(data_sandy, "sandy", tokenizer, en_stop, p_stemmer)
		print("Length of Data: {length}".format(length=len(data_sandy)))
		with open("storm_extracts/dict_sandy", "wb") as fp:   #Pickling
			pickle.dump(dict_sandy, fp)
		with open("storm_extracts/counts_sandy", "wb") as fp:   #Pickling
			pickle.dump(counts_sandy, fp)
	
	data_unt = read_data("unt")
	try:
		with open("storm_extracts/dict_unt", 'rb') as f:
			dict_unt = pickle.load(f)
		with open("storm_extracts/counts_unt", 'rb') as f:
			counts_unt = pickle.load(f)
	except:
		dict_unt, _, counts_unt = parse_text(data_unt, "unt", tokenizer, en_stop, p_stemmer)
		print("Length of Data: {length}".format(length=len(data_unt)))
		with open("storm_extracts/dict_unt", "wb") as fp:   #Pickling
			pickle.dump(dict_unt, fp)
		with open("storm_extracts/counts_unt", "wb") as fp:   #Pickling
			pickle.dump(counts_unt, fp)


	h_and_n = []
	s_and_n = []
	h_and_s = []
	u_and_h = []
	u_and_n = []
	u_and_s = []
	for num_topics in range(1, 6):
		lda_storm = run_model(data_storm, args.storm, num_topics)
		lda_noise = run_model(data_noise, "noise", num_topics)
		lda_sandy = run_model(data_sandy, "sandy", num_topics)
		lda_unt = run_model(data_unt, "unt", num_topics)

		print("~~~~~~~~~~~~~~~~~~~~~~~~~COMPARING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

		unweighted_h_and_n, weighted_h_and_n = compare_models(lda_storm, lda_noise, counts_storm, counts_noise, dict_storm, dict_noise, num_topics)
		unweighted_h_and_s, weighted_h_and_s = compare_models(lda_storm, lda_sandy, counts_storm, counts_sandy, dict_storm, dict_sandy, num_topics)
		unweighted_s_and_n, weighted_s_and_n = compare_models(lda_sandy, lda_noise, counts_sandy, counts_noise, dict_sandy, dict_noise, num_topics)

		unweighted_u_and_h, weighted_u_and_h = compare_models(lda_unt, lda_storm, counts_unt, counts_storm, dict_unt, dict_storm, num_topics)
		unweighted_u_and_s, weighted_u_and_s = compare_models(lda_unt, lda_sandy, counts_unt, counts_sandy, dict_unt, dict_sandy, num_topics)
		unweighted_u_and_n, weighted_u_and_n = compare_models(lda_unt, lda_noise, counts_unt, counts_noise, dict_unt, dict_noise, num_topics)

		h_and_n.append(weighted_h_and_n)
		h_and_s.append(weighted_h_and_s)
		s_and_n.append(weighted_s_and_n)

		u_and_h.append(weighted_u_and_h)
		u_and_s.append(weighted_u_and_s)
		u_and_n.append(weighted_u_and_n)

	print("harvey and noise")
	for i in range(len(h_and_n)):
		print("k="+str(i+1), h_and_n[i])
	print("sandy and noise")
	for i in range(len(s_and_n)):
		print("k="+str((i+1)/2), s_and_n[i])
	print("harvey and sandy")
	for i in range(len(h_and_s)):
		print("k="+str((i+1)/2), h_and_s[i])
	print("unt and harvey")
	for i in range(len(u_and_h)):
		print("k="+str(i+1), u_and_h[i])
	print("unt and noise")
	for i in range(len(u_and_n)):
		print("k="+str(i+1), u_and_n[i])	
	print("unt and sandy")
	for i in range(len(u_and_s)):
		print("k="+str(i+1), u_and_s[i])


if __name__ == '__main__':
	main()