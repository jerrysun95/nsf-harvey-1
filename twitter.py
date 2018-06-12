import json, os, ast, gzip, keyring, requests, random
import google_vision as gv
import bz2

# Grabs relevant information from a tweet and returns result
# If the tweet does not contain an image, returns None
def parse_tweet(tweet):
	d = {}
	if 'entities' in tweet:
		if 'media' in tweet['entities']:
			if 'media_url' in tweet['entities']['media'][0]:
				d['text'] = tweet['text']
				d['id'] = tweet['id']
				d['timestamp_ms'] = tweet['timestamp_ms']
				d['coordinates'] = tweet['coordinates']
				d['media_url'] = tweet['entities']['media'][0]['media_url']
				d['hashtags'] = [x['text'] for x in tweet['entities']['hashtags']]
				return d
	return None

# Unzips and parses tweets in all .gz files in directory denoted by path
# Writes results to tweets_with_links.json
def parse_tweets(path):
	tweets_with_links = []
	for filename in os.listdir(path):
		if '.gz' in filename:
			file = gzip.open(path + '/' + filename.strip(' \t\n\r'), 'rb')
			# for line in file:
			# 	tweet = ast.literal_eval(line)
			# 	d = parse_tweet(tweet)
			# 	if d != None:
			# 		tweets_with_links.append(d)
		elif '.bz2' in filename:
			with bz2.BZ2File(path + '/' + filename) as file:
				# for line in file.readlines():
				line = file.readline()
				while line:
					print('hello')
					tweet = ast.literal_eval(line)
					d = parse_tweet(tweet)
					if d != None:
						tweets_with_links.append(d)

					line = file.readline()
	with open('output/tweets_with_links_s3.json', 'w') as f:
		f.write(json.dumps(tweets_with_links, indent=4))

# Retrieves image data from an image url
def get_image_from_url(path, out):
	labels = []
	with open(path, 'r') as f:
		data = json.loads(f.read())
		for obj in data:
			print(obj)
			imageLabels = {}
			image = requests.get(obj['media_url']).content
			try:
				imageLabels = gv.vision_from_data_image(str(obj['id']), image)
				labels.append(imageLabels)
			except:
				pass
	with open(out, 'w') as g:
		g.write(json.dumps(labels, indent=4))

def get_random_sample(path, num_tweets, out):
	print('Getting random sample')
	tweets = []
	files = [x for x in os.listdir(path) if '.bz2' in x]
	num_samples = 10
	samples = random.sample(range(len(files)), num_samples)

	for s in samples:
		print('Sampling zip ' + str(s))
		file = files[s]
		count = 0
		with bz2.BZ2File(path + '/' + file) as f:
			line = f.readline()
			while line and count < num_tweets:
				tweet = ast.literal_eval(line)
				d = parse_tweet(tweet)
				if d != None:
					tweets.append(d)
					count += 1

				line = f.readline()

	res = random.sample(tweets, num_tweets)
	with open(out, 'w') as f:
		f.write(json.dumps(res, indent=4))

# get_image_from_url()

# parse_tweets('tweets')
# get_random_sample('tweets', 10, 'tweets/random_sample_test.json')
