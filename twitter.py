import json, os, ast, gzip, keyring, requests
import google_vision as gv

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
			for line in file:
				tweet = ast.literal_eval(line)
				d = parse_tweet(tweet)
				if d != None:
					tweets_with_links.append(d)
	with open('output/tweets_with_links.json', 'w') as f:
		f.write(json.dumps(tweets_with_links, indent=4))

# Retrieves image data from an image url
def get_image_from_url():
	labels = []
	with open('output/tweets_with_links.json') as f:
		data = json.load(f.read())
		for obj in data:
			print(obj)
			imageLabels = {}
			image = requests.get(obj['media_url']).content
			imageLabels = gv.vision_from_data_image(str(obj['id']), image)
			labels.append(imageLabels)
	with open('output/tweets_gv.json', 'w') as g:
		g.write(json.dumps(labels, indent=4))

get_image_from_url()

# parse_tweets('tweets')
