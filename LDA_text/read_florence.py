import pickle
import json

pre_landfall = []
landfall = []
aftermath = []
recovery = []

files = glob.glob('data/{STORM}/*'.format(STORM=storm))
count = 0
num_lines = 5483030
for filename in files:
    try:
        with open(filename, 'r') as f:
            print("Extracting file {FILE}".format(FILE=filename))
            headers = next(f).split(',')
            idx = headers.index('title')
            idx2 = headers.index('description')
            d_idx = headers.index('pubdate')
            for line in f:
                tweets = line.split(',')
                date_parts = tweets[d_idx].split(" ")[0].split("-")
                month, date = int(date_parts[1]), int(date_parts[2])

                if month == 8:
                    pre_landfall.append(tweets[idx])
                    pre_landfall.append(tweets[idx2])
                elif month == 9
                    if date <= 13:
                        pre_landfall.append(tweets[idx])
                        pre_landfall.append(tweets[idx2])
                    elif date <= 18:
                        landfall.append(tweets[idx])
                        landfall.append(tweets[idx2])
                    else date <= 21:
                        aftermath.append(tweet_dict["text"])
                    else:
                        recovery.append(tweet_dict["text"])
                else:
                    recovery.append(tweet_dict["text"])

                count += 1
                print('PROGRESS: ({c}%)'.format(c=(int(count / num_lines * 100))))
    except:
        print("ERROR: Extracting file {FILE} failed".format(FILE=filename))

print(len(pre_landfall))
print(len(landfall))
print(len(aftermath))
print(len(recovery))
print(num_lines)

with open("storm_extracts/f_pre_landfall", "wb") as fp:   #Pickling
        pickle.dump(pre_landfall, fp)
with open("storm_extracts/f_landfall", "wb") as fp:   #Pickling
        pickle.dump(landfall, fp)
with open("storm_extracts/f_aftermath", "wb") as fp:   #Pickling
        pickle.dump(aftermath, fp)
with open("storm_extracts/f_recovery", "wb") as fp:   #Pickling
        pickle.dump(recovery, fp)
