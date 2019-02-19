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

	data_storm = read_data(args.storm)
	print("Length of Data: {length}".format(length=len(data_storm)))

	data_noise = read_data("noise")
	print("Length of Data: {length}".format(length=len(data_noise)))

	for num_topics in range(1, 11):
		lda_storm = run_model(data_storm, args.storm, num_topics)
		lda_noise = run_model(data_noise, "noise", num_topics)

if __name__ == '__main__':
	main()