import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords, wordnet
from gensim import corpora, models
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
import numpy
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import gensim
from gensim.test.utils import datapath
import operator
import json
import time
import glob
import re
import ast

from text import *

all_words = {}

nltk.download('stopwords')
en_stop = stopwords.words('english')

def main():
    # storm = 'private'
    # data = read_data(storm)
    # print("Length of Data: {length}".format(length=len(data)))''
    num_topics = 3
    tokenizer = RegexpTokenizer(r'[a-z0-9\']+')
    p_stemmer = PorterStemmer()

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running Sandy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data_sandy = read_data("sandy") 
    try:
        with open("storm_extracts/dict_sandy", 'rb') as f:
            dict_sandy = pickle.load(f)
        with open("storm_extracts/counts_sandy", 'rb') as f:
            counts_sandy = pickle.load(f)
    except:
        dict_sandy, _, counts_sandy = parse_text(data_sandy, "sandy", tokenizer, en_stop, p_stemmer)
        print("Length of Data: {length}".format(length=len(data_sandy)))
        with open("storm_extracts/dict_sandy", "wb") as fp:   #Pickling
            pickle.dump(dict_sandy, fp)
        with open("storm_extracts/counts_sandy", "wb") as fp:   #Pickling
            pickle.dump(counts_sandy, fp)
    lda_sandy = run_model(data_sandy, "sandy", num_topics)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running Harvey ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data_harvey = read_data("harvey")
    try:
        with open("storm_extracts/dict_storm", 'rb') as f:
            dict_harvey = pickle.load(f)
        with open("storm_extracts/counts_storm", 'rb') as f:
            counts_harvey = pickle.load(f)
    except:
        dict_storm, _, counts_storm = parse_text(data_storm, "harvey", tokenizer, en_stop, p_stemmer)
        print("Length of Data: {length}".format(length=len(data_storm)))
        with open("storm_extracts/dict_storm", "wb") as fp:   #Pickling
            pickle.dump(dict_harvey, fp)
        with open("storm_extracts/counts_storm", "wb") as fp:   #Pickling
            pickle.dump(counts_harvey, fp)
    lda_harvey = run_model(data_harvey, "harvey", num_topics)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running Florence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data_Florence = read_data("Florence")
    dict_Florence, _, counts_Florence = parse_text(data_Florence, "Florence", tokenizer, en_stop, p_stemmer)
    lda_florence = run_model(data_Florence, "Florence", num_topics)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running Lane ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data_Lane = read_data("Lane")
    dict_Lane, _, counts_Lane = parse_text(data_Lane, "Lane", tokenizer, en_stop, p_stemmer)
    lda_lane = run_model(data_Lane, "Lane", num_topics)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running Michael ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data_Michael = read_data("Michael")
    dict_Michael, _, counts_Michael = parse_text(data_Michael, "Michael", tokenizer, en_stop, p_stemmer)
    lda_michael = run_model(data_Michael, "Michael", num_topics)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running Bonnie ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data_bonnie = read_data("bonnie")
    dict_bonnie, _, counts_bonnie = parse_text(data_bonnie, "bonnie", tokenizer, en_stop, p_stemmer)
    lda_bonnie = run_model(data_bonnie, "bonnie", num_topics)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running private ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data_private= read_data("private")
    dict_private, _, counts_private = parse_text(data_private, "private", tokenizer, en_stop, p_stemmer)
    lda_private = run_model(data_private, "private", num_topics)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Running noise ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    data_noise = read_data("noise")
    try:
        with open("storm_extracts/dict_noise", 'rb') as f:
            dict_noise = pickle.load(f)
        with open("storm_extracts/counts_noise", 'rb') as f:
            counts_noise = pickle.load(f)
    except:
        dict_noise, _, counts_noise = parse_text(data_noise, "noise", tokenizer, en_stop, p_stemmer)
        print("Length of Data: {length}".format(length=len(data_noise)))
        with open("storm_extracts/dict_noise", "wb") as fp:   #Pickling
            pickle.dump(dict_noise, fp)
        with open("storm_extracts/counts_noise", "wb") as fp:   #Pickling
            pickle.dump(counts_noise, fp)
    lda_noise = run_model(data_noise, "noise", num_topics)



    models = [lda_sandy, lda_harvey, lda_florence, lda_lane, lda_michael, lda_bonnie, lda_private, lda_noise]
    model_names = ["lda_sandy", "lda_harvey", "lda_florence", "lda_lane", "lda_michael", "lda_bonnie", "lda_private", "lda_noise"]

    print("Printing sorted scores per model...")
    scores = []
    for i in range(len(model_names)):
        print("COMPARING DATASETS TO: {a}".format(a=model_names[i]))
        per_model = []
        for j in range(len(model_names)):
            if i == j:
                continue
            # print("Comparing {a} with {b}".format(a=model_names[i], b=model_names[j]))
            dist = compare_models(models[i], models[j])
            scores.append((i, j, dist))
            per_model.append((i, j, dist))

        per_model = sorted(per_model, key=operator.itemgetter(2))
        for i, j, dist in per_model:
            print("Comparing {a} with {b}".format(a=model_names[i], b=model_names[j]))
            print("Hellinger distance:", dist)
        print("\n")

    print("Printing total sorted scores...")
    scores = sorted(scores, key=operator.itemgetter(2))
    for i, j, dist in scores:
        print("Comparing {a} with {b}".format(a=model_names[i], b=model_names[j]))
        print("Hellinger distance:", dist)


    # top = top_words(models[:-1])
    # for word, count in top:
    #   print(str(count) + "\t" + word)

    # for i in range(len(model_names)):
    #   print(model_names[i])
    #   top = top_words(models[i:i+1])
    #   for word, count in top[:10]:
    #       print(word, count)
    #   print("\n")

    # visualize(lda, dictionary, corpus)

    # bow = dictionary.doc2bow(["Help", "i", "am", "stuck", "in", "a", "storm"])
    # print(bow)
    # topics = lda.get_document_topics(bow, per_word_topics=True)
    # print("TOPICS-S:", topics)

    # bow = dictionary.doc2bow(["check", "out", "my", "new", "album"])
    # print(bow)
    # topics = lda.get_document_topics(bow, per_word_topics=True)
    # print("TOPICS-N:", topics)



def read_data(storm):
    print("Extracting data...")
    data = []
    start = time.time()

    try:
        with open("storm_extracts/{storm}".format(storm=storm), 'rb') as f:
            data = pickle.load(f)

        print("Data already extracted.")
        end = time.time()
        print("Time taken to run: {SEC} seconds".format(SEC=end - start))
        return data
    except:
        pass

    if storm == 'sandy':
        with open('data/tweets_sandy/affected_tweets_clean.pkl', 'rb') as f:
            data.extend(pickle.load(f))

        with open('data/tweets_sandy/destroyed_tweets_clean.pkl', 'rb') as f:
            data.extend(pickle.load(f))

        with open('data/tweets_sandy/major_tweets_clean.pkl', 'rb') as f:
            data.extend(pickle.load(f))

        with open('data/tweets_sandy/minor_tweets_clean.pkl', 'rb') as f:
            data.extend(pickle.load(f))

    elif storm == 'harvey':
        files = glob.glob('data/tweets_harvey/tweets.log*')
        for filename in files:
            print("Extracting file {FILE}".format(FILE=filename))
            f = open(filename, 'r').read()
            indicesStart = [m.start() for m in re.finditer('(?=text.*is_quote_status)', f)]
            indicesEnd = [m.start() for m in re.finditer('(?=is_quote_status)', f)]

            iS = 0
            iE = 0

            while (iS+1 < len(indicesStart) and iE < len(indicesEnd)):
                if indicesStart[iS+1] > indicesEnd[iE]:
                    data.append(f[indicesStart[iS]+9:indicesEnd[iE]-5])
                    iE += 1
                else:
                    iS += 1

            if iE < len(indicesEnd):
                data.append(f[indicesStart[iS]+9:indicesEnd[iE]-5])
    elif storm == 'Florence' or storm == 'Lane' or storm == 'Michael':
        files = glob.glob('data/{STORM}/*'.format(STORM=storm))
        for filename in files:
            try:
                with open(filename, 'r') as f:
                    print("Extracting file {FILE}".format(FILE=filename))
                    headers = next(f).split(',')
                    idx = headers.index('title')
                    idx2 = headers.index('description')
                    for line in f:
                        tweets = line.split(',')
                        data.append(tweets[idx])
                        data.append(tweets[idx2])
            except:
                print("ERROR: Extracting file {FILE} failed".format(FILE=filename))
    elif storm == 'bonnie':
        filename = "data/bonnie.csv"
        with open(filename, 'r') as f:
                print("Extracting file {FILE}".format(FILE=filename))
                while True:
                    try:
                        line = next(f)
                        print(line) 
                        data.append(line)
                    except StopIteration:
                        break
                    except:
                        print("skip")
    elif storm == 'private':
        files = glob.glob('data/{STORM}/*'.format(STORM=storm))
        for filename in files:
            with open(filename, 'r') as f:
                print("Extracting file {FILE}".format(FILE=filename))
                while True:
                    try:
                        line = next(f)  
                        data.append(line)
                    except StopIteration:
                        break
                    except:
                        print("skip")
    else:
        files = glob.glob('data/tweets_noise/tweets_random*')
        for f in files[:len(files)]:
            print(f)
            noise = json.load(open(f))
            for j in noise:
                j = j["text"].lower()
                data.append(j)

    # for sentence in data:
    #   for word in sentence.split(" "):
    #       if word in all_words:
    #           all_words[word] += 1
    #       else:
    #           all_words[word] = 1


    with open("storm_extracts/{storm}".format(storm=storm), "wb") as fp:   #Pickling
        pickle.dump(data, fp)

    end = time.time()
    print("Length of data: {l}".format(l=len(data)))
    print("Time taken to run: {SEC} seconds".format(SEC=end - start))

    return data

def run_model(data, storm, num_topics=5, print_flag=False):
    tokenizer = RegexpTokenizer(r'[a-z0-9\']+')
    p_stemmer = PorterStemmer()

    model = "{storm}_{num}".format(storm=storm, num=num_topics)
    start = time.time() 
    save_file = datapath(model)
    try:
        # try:
        lda = gensim.models.ldamodel.LdaModel.load(save_file)
        # except FileNotFoundError:
        #     lda = gensim.models.ldamulticore.LdaMulticore.load(save_file)
    except:
        print("WARNING: Model not found...")
        dictionary, corpus, counts = parse_text(data, model, tokenizer, en_stop, p_stemmer)
        
        lda = build_model(dictionary, corpus, num_topics)
        lda.save(save_file)

    end = time.time()
    # print("Time taken to run: {SEC} seconds".format(SEC=end - start))

    if print_flag:
        print("Printing topics...")
        topics = lda.show_topics(formatted=False, log=False)
        res = []
        for t in topics:
            print("NEW TOPIC...")
            for word in t:
                print(word)
            res.append(t)

    return lda


def update(data, dictionary):
    print("BEFORE UPDATE...")
    dictionary, new_corpus = parse_text(data, "affected_tweets_clean-2", tokenizer, en_stop, p_stemmer, dictionary=dictionary)
    lda.update(new_corpus)

    bow = dictionary.doc2bow(data)
    print(bow)
    # topics = lda.show_topics(num_topics=5, num_words=15, formatted=False, log=False)
    topics = lda.get_document_topics(bow, per_word_topics=True)

    res = []
    for t in topics:
        print("NEW TOPIC...")
        for word in t:
            print(word)
        res.append(t)

def compare(lda, corpus, lda2, corpus2):
    #Train your LDA model    
    # lda = LdaModel(corpus, num_topics=10)

    # Get the mean of all topic distributions in one corpus
    storm_topic_vectors = []
    for d in corpus:
        storm_topic_vectors.append(lda[d])
    storm_average = numpy.average(numpy.array(storm_topic_vectors), axis=0)

    # Get the mean of all topic distributions in another corpus
    other_topic_vectors = []
    for d in corpus2:
        other_topic_vectors.append(lda[d])
    other_average = numpy.average(numpy.array(other_topic_vectors), axis=0)

    # Calculate the distance between the distribution of topics in both corpora
    difference_of_distributions = numpy.linalg.norm(storm_average - other_average)

    # Hellinger distance
    # KL divergence

def make_topics_bow(topic, all_words):
    for i in range(len(topic)):
        # print(topic[i][0])
        word = topic[i][0]
        topic[i] = (all_words.get_index(word), topic[i][1])
    return topic
        
def compare_models(lda1, lda2, counts1, counts2, d1, d2, num_topics):
    difference, anno = lda1.diff(lda2, distance="jaccard", normed=False)
    # print (anno)
    # difference = 1 - difference
    total_counts_1 = sum(counts1.values())
    total_counts_2 = sum(counts2.values())

    weights = np.zeros((num_topics, num_topics))
    total_percentage = 0

    for i in range(num_topics):
        t1 = lda1.get_topic_terms(i)
        t2 = lda2.get_topic_terms(i)

        t1_count = 0
        for term_id, _ in t1:
            term = d1[term_id]
            t1_count += counts1[term]
        percentage1 = t1_count/total_counts_1

        t2_count = 0
        for term_id, _ in t2:
            term = d2[term_id]
            t2_count += counts2[term]
        percentage2 = t2_count/total_counts_2

        total_percentage += percentage1 + percentage2

        
        for j in range(num_topics):
            # add to row i percentage1
            weights[i][j] += percentage1
            # add to column i percentage 2
            weights[j][i] += percentage2


    # divide entire matrix by total_percentage
    weights /= total_percentage
    difference *= 100

    # find similar words for most weighted topic
    # max_weight = weights[0][0]
    # mr, mc = 0, 0
    # for r in range(np.size(weights,0)):
    #   for c in range(np.size(weights,1)):
    #       if weights[r][c] > max_weight:
    #           mr, mc = r, c
    #           max_weight = weights[r][c]

    # print(anno[mr][mc][0]) 
    # print("difference", difference)
    # print("WEIGHTS", weights)
    # print("diff*weights", difference*weights)
    # print("\n")

    return [(difference).sum(), (difference*weights).sum()]

def top_words(models):
    words = {}
    for model in models:
        for topics in model.show_topics(formatted=False):
            for topic in topics[1]:
                # for pairs in topic[1]:
                w = topic[0]
                if w in words:
                    words[w] += 1
                else:
                    words[w] = 1
    words = sorted(words.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted(all_words.items(), key=operator.itemgetter(1), reverse=True)[0:100])
    return words

class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    # Returns -1 if the object isn't present, index otherwise
    def index_of(self, object):
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    # Adds the object to the index if it isn't present, always returns a nonnegative index
    def get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]



if __name__ == '__main__':
    main()