"""
Helpers borrowed from baseline implementation in feature_engineering.py
"""

import re
import nltk
from nltk import tokenize
import numpy as np
from sklearn import feature_extraction
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
def get_body(n, df):
    return df.loc[lambda x: x["Body ID"] == n, "articleBody"].item()

"""
input: string w/ multiple sentences
output: dictionary object with sentiment scores for each category

average of sentiment scores for each sentence
TODO: add weightings for sentences based on importance
"""
def sentiment_multi(paragraph):
    categories = ["pos", "neg", "neu", "compound"]
    sentence_list = tokenize.sent_tokenize(paragraph)
    paragraphSentiments = {"pos":0, "neg":0, "neu":0, "compound":0}
    for sentence in sentence_list:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<69} {}".format(sentence, str(vs["compound"])))
        for cat in categories:
            paragraphSentiments[cat] += vs[cat]
    for cat in categories:
        paragraphSentiments[cat] /= len(sentence_list)
    return paragraphSentiments