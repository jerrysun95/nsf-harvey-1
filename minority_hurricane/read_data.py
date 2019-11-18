import pickle
import json
import os
import glob

# this is to read all the terms 
terms = open('terms.csv',encoding ="utf-8-sig")
terms_dict = {}
for line in terms:
	line=line.lower()
	words = line.strip().split(',')
	words = [word.strip() for word in words if len(word) > 0]
	terms_dict[words[0]] = words[1:]

print(terms_dict)

relevant_tweets = {race: set() for race in terms_dict}


data_zipped = './Data'
data_unzipped = './data_unzipped2'

files = os.listdir(data_zipped)

# for filename in files:
# 	os.system("gunzip -c " + data_zipped+'/'+filename + ' > ' + data_unzipped+'/'+filename[:-3])

files = os.listdir(data_unzipped)

for filename in files:
	print("Extracting file {FILE}".format(FILE=filename))
	f = open(data_unzipped+'/'+filename,"r")
	for line in f:
		tweet = json.loads(line)
		try:
			text = tweet["text"].lower()
			for race in terms_dict:
				for word in terms_dict[race]:
					if word in text:
						# print(word, '~~~~~~~',text)
						relevant_tweets[race].add(text)

		except KeyError:
			pass

relevant_tweets = {k: list(relevant_tweets[k]) for k in relevant_tweets}

print(relevant_tweets["hispanic"])

with open("relevant_tweets.json","w") as outfile:
	json.dump(relevant_tweets,outfile)

print("Done")

