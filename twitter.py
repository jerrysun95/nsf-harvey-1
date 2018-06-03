import json, os, ast, gzip, keyring, requests

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
	with open('tweets_with_links.json', 'w') as f:
		f.write(json.dumps(tweets_with_links, indent=4))

# Retrieves image data from an image url
def get_image_from_url (url, filename):
    image = requests.get(url).content

    access_token = keyring.get_password("system", "BOX_ACCESS_TOKEN")
    print(access_token)
    parent_id = 0

    headers = { 'Authorization' : 'Bearer {0}'.format(access_token) }
    url = 'https://upload.box.com/api/2.0/files/content'
    files = { 'filename': (filename, image) }
    data = { "parent_id": parent_id }
    response = requests.post(url, data=data, files=files, headers=headers)
    file_info = response
    print(file_info)

# get_image_from_url('https://pbs.twimg.com/media/DJ4D7ZJV4AA_Zsi.jpg', 'test')

parse_tweets('tweets')
