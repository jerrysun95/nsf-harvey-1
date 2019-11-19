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
def wordFound(sword,user_Description):
	for word in sword:
		if word not in user_Description:
			return False
	return True

for filename in files:
	print("Extracting file {FILE}".format(FILE=filename))
	f = open(data_unzipped+'/'+filename,"r")
	for line in f:
		tweet = json.loads(line)
		try:

			text = set(tweet["text"].lower().split())

			user = tweet["user"]["description"]
			if user !=None and user!="":
				user = set(user.lower().split())
			else:
				user=""

			for race in terms_dict:
				for word in terms_dict[race]:
					sword=word.split()
					if wordFound(sword,user):
						relevant_tweets[race].add(tweet["user"]["description"].lower())

					if wordFound(sword,text):
						relevant_tweets[race].add(tweet["text"].lower())

		except KeyError:
			pass

relevant_tweets = {k: list(relevant_tweets[k]) for k in relevant_tweets}
# for race in terms_dict:
# 	print(len(relevant_tweets[race])
# print(relevant_tweets["hispanic"])

with open("relevant_tweets.json","w") as outfile:
	json.dump(relevant_tweets,outfile)

print("Done")
for x in relevant_tweets:
	print(len(relevant_tweets[x]))