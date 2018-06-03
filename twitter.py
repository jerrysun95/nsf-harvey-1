#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import json, re, os, ast, gzip


# print(len(d))

# def load_dirty_json(dirty_json):
#     regex_replace = [('\\U', '\\\\U'), (r" None([, \}\]])", r' null\1'), (r'"', '\\"'), (r"([ \{,:\[])(u)?'([^']+)'", r'\1"\3"'), (r" False([, \}\]])", r' false\1'), (r" True([, \}\]])", r' true\1')]
#     for r, s in regex_replace:
#         dirty_json = re.sub(r, s, dirty_json)
#     with open('tweets/json_fun.json', 'w') as f:
#     	f.write(dirty_json)
#     clean_json = json.loads(dirty_json)
#     return clean_json

# with open('tweets/tweets.log-2017-09-17-20-00-01_harvey') as f:
# 	j = load_dirty_json(f.read())

def parse_tweet(tweet):
	d = {}
	# print(tweet)
	if 'entities' in tweet:
		# print('entities')
		# print(tweet['entities'].keys())
		if 'media' in tweet['entities']:
			# print('media')
			# print(tweet['entities']['media'].keys())
			if 'media_url' in tweet['entities']['media'][0]:
				# print('media_url')
				d['text'] = tweet['text']
				d['id'] = tweet['id']
				d['timestamp_ms'] = tweet['timestamp_ms']
				d['coordinates'] = tweet['coordinates']
				d['media_url'] = tweet['entities']['media'][0]['media_url']
				d['hashtags'] = [x['text'] for x in tweet['entities']['hashtags']]
				return d
	return None

def parse_tweets():
	tweets_with_links = []
	for filename in os.listdir('tweets'):
		if '.gz' in filename:
			file = gzip.open('tweets/' + filename.strip(' \t\n\r'), 'rb')
			for line in file:
				tweet = ast.literal_eval(line)
				d = parse_tweet(tweet)
				if d != None:
					tweets_with_links.append(d)
	with open('tweets_with_links.json', 'w') as f:
		f.write(json.dumps(tweets_with_links, indent=4))

parse_tweets()
