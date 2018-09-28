"""
Helpers borrowed from baseline implementation in feature_engineering.py
"""

import re
import nltk
from nltk import tokenize
import numpy as np
from sklearn import feature_extraction
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter

_wnl = nltk.WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

"""
stolen from SO: https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
i/o: string representing POS tag
"""
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

"""
imput: string
output: string
"""
def normalize_word(w):
    return _wnl.lemmatize(w).lower()

"""
input: string
output: string list
"""
def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]

"""
imput: string
output: string
"""
def clean_lower(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

"""
imput: string
output: string
"""
def clean(s):
    # Cleans a string: trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE))

"""
input: string list
output: string list
"""
def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

"""
imput: string
output: string list
"""
def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

"""
input: row number, dataframe (train_stances)
output: string of headline
"""
def get_head(n, df):
    return (df.iloc[n])["Headline"]

"""
input: body ID, dataframe (train_bodies)
output: string of body text
"""
# get_body(6, train_bodies)
def get_body(n, df):
    return df.loc[lambda x: x["Body ID"] == n, "articleBody"].item()

"""
outputs sentiment for single sentence
"""
def get_sentiment(sentence):
    return analyzer.polarity_scores(sentence)

"""
input: string w/ multiple sentences
output: dictionary object with sentiment scores for each category

average of sentiment scores for each sentence
TODO: add weightings for sentences based on importance
{"pos", "neg", "neu", "compound"}
"""
def sentiment_multi(paragraph):
    categories = ["pos", "neg", "neu", "compound"]
    sentence_list = tokenize.sent_tokenize(paragraph)
    paragraphSentiments = {"pos":0, "neg":0, "neu":0, "compound":0}
    for sentence in sentence_list:
        vs = get_sentiment(sentence)
        for cat in categories:
            paragraphSentiments[cat] += vs[cat]
    for cat in categories:
        paragraphSentiments[cat] /= len(sentence_list)
    return paragraphSentiments

"""
builds a vocabulary from the given column in the data frame
input: df - dataframe object, col - string column name, 
pos_tags - optional, list of pos tags to include (default: include all tokens)
output: list of strings
"""
# example usage
# build_vocabulary(train_stances, 'Headline', ['NN','NNS','NNP','NNPS'])

def build_vocabulary(df, col, pos_tags = None):
    vocabulary = set()
    for body in list(df[col]):
        clean_body = clean(body)
        tokens = get_tokenized_lemmas(clean_body)
        clean_tokens = remove_stopwords(tokens)
        pos = nltk.pos_tag(clean_tokens)
        if pos_tags is None:
            filtered_tokens = clean_tokens
        else:
            filtered_tokens = [x[0] for x in pos if (x[1] in pos_tags)]
        vocabulary = vocabulary.union(set(nouns))
    return list(vocabulary)

"""
extract metadata from sentence/body of text
input: string
output: dict
"""
def process_sentence(s):
    clean_s = clean(s)
    tokens = get_tokenized_lemmas(clean_s)
    clean_tokens = remove_stopwords(tokens)
    
    bigram = list(nltk.bigrams(clean_tokens))
    bigram_str = [x[0]+' '+x[1] for x in bigram]
    trigram = list(nltk.trigrams(clean_tokens))
    trigram_str = [x[0]+' '+x[1]+' '+x[2] for x in trigram]
    
    pos = nltk.pos_tag(clean_tokens)
    #count of each tag type (dict)
    tags_count = Counter([x[1] for x in pos])
    
    #list of words that belong to that part of speech
    nouns = [x[0] for x in pos if is_noun(x[1])]
    verbs = [x[0] for x in pos if is_verb(x[1])]
    adjectives = [x[0] for x in pos if is_adjective(x[1])]
    adverbs = [x[0] for x in pos if is_adverb(x[1])]
    
    vader_sentiment = get_sentiment(s)

    return {
        "tokens": clean_tokens,
        "bigrams": bigram_str,
        "trigrams": trigram_str,
        "pos_count": tags_count,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "sentiment": vader_sentiment
    }

# example usage:
# process_body(get_body(6, train_bodies))['pos_count']
def process_body(body):
    clean_body = clean(body)
    tokens = get_tokenized_lemmas(clean_body)
    clean_tokens = remove_stopwords(tokens)
    
    #look at first sentence of article
    first_sentence = list(nltk.tokenize.sent_tokenize(body))[0]
    first_sentence_data = process_sentence(first_sentence)
    
    bigram = list(nltk.bigrams(clean_tokens))
    bigram_str = [x[0]+' '+x[1] for x in bigram]
    trigram = list(nltk.trigrams(clean_tokens))
    trigram_str = [x[0]+' '+x[1]+' '+x[2] for x in trigram]
    
    pos = nltk.pos_tag(clean_tokens)
    #count of each tag type (dict)
    tags_count = Counter([x[1] for x in pos])
    
    #list of words that belong to that part of speech
    nouns = [x[0] for x in pos if is_noun(x[1])]
    verbs = [x[0] for x in pos if is_verb(x[1])]
    adjectives = [x[0] for x in pos if is_adjective(x[1])]
    adverbs = [x[0] for x in pos if is_adverb(x[1])]
    
    #breakdown of porportion of regular/comparative/superlative adverbs as a tuple (in that order)
    adj_types = (tags_count['JJ']/len(adjectives),tags_count['JJR']/len(adjectives),tags_count['JJS']/len(adjectives))
    adv_types = (tags_count['RB']/len(adjectives),tags_count['RBR']/len(adjectives),tags_count['RBS']/len(adjectives))

    vader_sentiment = sentiment_multi(body)
    
    return {
        "tokens": clean_tokens,
        "bigrams": bigram_str,
        "trigrams": trigram_str,
        "pos_count": tags_count,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "sentiment": vader_sentiment,
        "first_sentence": first_sentence_data,
        "adj_types": adj_types,
        "adv_types": adv_types
    }

"""
in: bodies dataframe with Body ID and articleBody columns
out: dict with k=bodyid and v=dict of bodyinfo as per process_body
"""
def process_bodies(df)
    body_info = {}
    ids = list(df["Body ID"])
    for i in range(len(ids)):
        if i%100 == 0 and i!=0:
            print("processed "+str(i))
        body_info[ids[i]]= process_body(get_body(6, df))
    print("done! processed "+ str(len(ids)))
    return body_info