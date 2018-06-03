import sys, os
from datetime import datetime
import codecs
import json
import gzip
import ast

def map_month(name):
	if name=='Jan':
		return 1
	elif name=='Feb':
		return 2
	elif name=='Mar':
		return 3
	elif name=='Apr':
		return 4
	elif name=='May':
		return 5
	elif name=='June':
		return 6
	elif name=='July':
		return 7
	elif name=='August':
		return 8
	elif name=='September':
		return 9
	elif name=='October':
		return 10
	elif name=='November':
		return 11
	else:
		return 12

def map_date(s):
	temp = s.split()
	year = int(temp[5])
	date = int(temp[2])
	month = map_month(temp[1])
	mini = temp[3].split(':')
	hour = int(mini[0].strip(' \t\n\r'))
	minute = int(mini[1].strip(' \t\n\r'))
	second = int(mini[2].strip(' \t\n\r'))
	p = datetime(year,month,date,hour,minute,second)
	return p
Date = ['2017-08-17','2017-08-18','2017-08-19','2017-08-20','2017-08-21','2017-08-22','2017-08-23','2017-08-24','2017-08-25','2017-08-26','2017-08-27','2017-08-28','2017-08-29','2017-08-30','2017-08-31','2017-09-01','2017-09-02','2017-09-03','2017-09-04','2017-09-05','2017-09-06','2017-09-07','2017-09-08','2017-09-09','2017-09-10','2017-09-11','2017-09-12','2017-09-13','2017-09-14','2017-09-15','2017-09-16','2017-09-17']
for d in Date:
	command = 'ls data/*' + d + '*.gz > temp.txt'
	os.system(command)
	ofname = 'Harvey_' + d + '.txt'
	fo = codecs.open(ofname,'w','utf-8')
	fs = open('temp.txt','r')
	for name in fs:
        	fp = gzip.open(name.strip(' \t\n\r'),'rb')
        	for l in fp:
                	dic = ast.literal_eval(l)
			#dic = json.loads(l.strip(' \t\n\r'))
			################################################ collect date, tweet_id, user_id, text ##############################################
                	temp = dic['text'].split()
                	text = ''
                	for x in temp:
                        	text = text + x.encode('ascii','ignore') + ' '
                	text = text.strip()
			dtm = map_date(dic['created_at'])
                	s = str(dtm) + '\t' + dic['id_str'] + '\t' + dic['user']['id_str'] + '\t' + dic['user']['screen_name'] + '\t' + text

			################################################# collect media (picture links) ######################################################
                	media = ''
                	temp = dic['entities']
                	if temp.__contains__('media')==True:
                        	for x in temp['media']:
					media = media + x['media_url'] + ' '
                	#print(temp.keys())
                	if dic.__contains__('quoted_status')==True:
                        	temp = dic['quoted_status']['entities']
                        	if temp.__contains__('media')==True:
                                	for x in temp['media']:
                                        	media = media + x['media_url'] + ' '
			if len(media)>1:
                		media = media.strip(' \t\n\r')
				s = s + '\t' + media
			else:
				s = s + '\t' + 'NULL'

			################################################## collect friends, followers, statuses count ###########################################
                	s = s + '\t' + str(dic['user']['friends_count']) + '\t' + str(dic['user']['followers_count']) + '\t' + str(dic['user']['statuses_count'])
	
			################################################## collect place information ############################################################
			if dic['place']==None:
				s = s + '\t' + 'NULL'
			else:
				#print(dic['place'])
				s = s + '\t' + dic['place']['name']

			################################################### collect language information #########################################################
			s = s + '\t' + dic['lang']

			################################################## collect retweet information ###########################################################
			if dic.__contains__('retweeted_status')==True:
				temp = dic['retweeted_status']
				rts = temp['id_str'] + '\t' + temp['user']['id_str'] + '\t' + temp['user']['screen_name'] + '\t' + str(temp['user']['followers_count'])
				if temp['place']==None:
					rts = rts + '\t' + 'NULL'
				else:
					rts = rts + '\t' + temp['place']['name']
			else:
				rts = 'NULL' + '\t' + 'NULL' + '\t' + 'NULL' + '\t' + 'NULL'
			s = s + '\t' + rts

			################################################### collect reply information #############################################################
			if dic['in_reply_to_status_id_str']==None:
				s = s + '\t' + 'NULL'
			else:
				s = s + '\t' + dic['in_reply_to_status_id_str']
                	fo.write(s + '\n')
        	fp.close()
	fs.close()
	fo.close()
	print('Complete: ',d)

