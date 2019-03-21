import json
import pickle

file_name = "/corral-repl/utexas/Trump-Tweets/Harvey/unt_data/harvey_twitter_dataset/02_archive_only/HurricaneHarvey.json"
count = 0.0
num_lines = 7041866
storm = 'unt'

pre_landfall = []
landfall = []
aftermath = []
recovery = []

with open(file_name) as f:
    for line in f:
        tweet_dict = json.loads(line)
        date_parts = tweet_dict["created_at"].split(" ")
        month, date = date_parts[1], int(date_parts[2])

        if month == "Aug":
            if date <= 24:
                pre_landfall.append(tweet_dict["text"])
            elif date <= 30:
                landfall.append(tweet_dict["text"])
            else:
                aftermath.append(tweet_dict["text"])
        else:
            if date <= 3:
                aftermath.append(tweet_dict["text"])
            else:
                recovery.append(tweet_dict["text"])

        # tweets.append(tweet_dict["text"])
        count += 1
        print('PROGRESS: ({c}%)'.format(c=(int(count / num_lines * 100))))

print(len(pre_landfall))
print(len(landfall))
print(len(aftermath))
print(len(recovery))
# print(len(tweets))

with open("storm_extracts/pre_landfall", "wb") as fp:   #Pickling
        pickle.dump(pre_landfall, fp)
with open("storm_extracts/landfall", "wb") as fp:   #Pickling
        pickle.dump(landfall, fp)
with open("storm_extracts/aftermath", "wb") as fp:   #Pickling
        pickle.dump(aftermath, fp)
with open("storm_extracts/recovery", "wb") as fp:   #Pickling
        pickle.dump(recovery, fp)

