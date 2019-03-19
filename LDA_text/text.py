from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from gensim import corpora, models
import gensim
import operator

# create sample documents
doc_a = 'The Super Bowl is the annual championship game of the National Football League (NFL). The game is the culmination of a regular season that begins in the late summer of the previous calendar year. Normally, Roman numerals are used to identify each game, rather than the year in which it is held. For example, Super Bowl I was played on January 15, 1967, following the 1966 regular season. The sole exception to this naming convention tradition occurred with Super Bowl 50, which was played on February 7, 2016, following the 2015 regular season, and the following year, the nomenclature returned to Roman numerals for Super Bowl LI, following the 2016 regular season. The most recent Super Bowl was Super Bowl LII, on February 4, 2018, following the 2017 regular season. The game was created as part of a merger agreement between the NFL and its then-rival league, the American Football League (AFL). It was agreed that the two leagues champion teams would play in the AFLNFL'

doc_b = 'World Championship Game until the merger was to officially begin in 1970. After the merger, each league was redesignated as a conference, and the game has since been played between the conference champions to determine the NFLs league champion. Currently, the National Football Conference (NFC) leads the league with 27 wins to 25 wins for the American Football Conference (AFC). The Pittsburgh Steelers have the most Super Bowl championship titles, with six. The New England Patriots have the most Super Bowl appearances, with ten. Charles Haley and Tom Brady both have five Super Bowl rings, which is the record for the most rings won by a single player. The day on which the Super Bowl is played, now considered by some as an unofficial American national holiday,[1][2] is called Super Bowl Sunday. It is the second-largest day for U.S. food consumption, after Thanksgiving Day.[3] In addition, the Super Bowl has frequently been the most-watched American television broadcast of the year'

doc_c = 'the seven most-watched broadcasts in U.S. television history are Super Bowls.[4] In 2015, Super Bowl XLIX became the most-watched American television program in history with an average audience of 114.4 million viewers, the fifth time in six years the game had set a record, starting with the 2010 Super Bowl, which itself had taken over the number-one spot held for 27 years by the final episode of M*A*S*H.[5][6][7] The Super Bowl is also among the most-watched sporting events in the world, almost all audiences being North American, and is second to soccers UEFA Champions League final as the most watched annual sporting event worldwide.[8]'

doc_d = 'The NFL restricts the use of its Super Bowl trademark; it is frequently called the Big Game or other generic terms by non-sponsoring corporations. Because of the high viewership, commercial airtime during the Super Bowl broadcast is the most expensive of the year, leading to companies regularly developing their most expensive advertisements for this broadcast. As a result, watching and discussing the broadcasts commercials has become a significant aspect of the event.[9] In addition, popular singers and musicians including Mariah Carey, Michael Jackson, Madonna, Prince, Beyonc, Paul McCartney, The Rolling Stones, The Who, Whitney Houston, and Lady Gaga have performed during the events pre-game and halftime ceremonies.'

# compile sample documents into a list
doc_set = [doc_a, doc_b, doc_c, doc_d]

# list for tokenized documents in loop

def parse_text(doc, name, tokenizer, en_stop, p_stemmer, dictionary=None):
    texts = []
    print('DOC NAME: ' + name + '\n')
    # loop through document list
    for i in range(len(doc)):
        raw = doc[i].lower()
        tokens = tokenizer.tokenize(raw)

        stopped_tokens = [tok for tok in tokens if not tok in en_stop]
        cleaned_tokens = [tok for tok in stopped_tokens if wordnet.synsets(tok)]
        stemmed_tokens = [p_stemmer.stem(tok) for tok in cleaned_tokens]
        sized_tokens = [tok for tok in stemmed_tokens if len(tok) > 2 and tok != 'http']

        # add tokens to list
        # print(sized_tokens)
        if len(sized_tokens) > 0:
            texts.append(sized_tokens)

    counts = {}
    for e in texts:
        for w in e:
            if w not in counts:
                counts[w] = 0
            counts[w] += 1
    # for word, count in sorted(counts.items(), key=operator.itemgetter(1), reverse=True)[0:20]:
    #     print(word, count)

    # turn our tokenized documents into a id <-> term dictionary
    if dictionary == None:
        dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, corpus, counts

def build_model(dictionary, corpus, num_topics):

    print("Generating model")
    # generate LDA model
    ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=num_topics, id2word = dictionary, passes=30, workers=3)

    # print("Printing topics")
    # topics = ldamodel.show_topics(num_topics=5, num_words=15, formatted=False, log=False)
    # res = []
    # for t in topics:
    #     print("NEW TOPIC")
    #     for word in t:
    #         print(word)
    #     res.append(t)
    return ldamodel

