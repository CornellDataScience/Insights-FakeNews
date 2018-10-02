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
stolen from SO: 
stackoverflow.com/questions/25534214

in: string representing POS tag
out: bool
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
takes word and lemmatizes/makes lowercase
in: string
out: string
"""
def normalize_word(w):
    return _wnl.lemmatize(w).lower()

"""
lemmatized, lowercased tokens
in: string
out: string list
"""
def get_tokenized_lemmas(s):
    lemmas = [normalize_word(t) for t in nltk.word_tokenize(s)]
    return [i for i in lemmas if len(i)>1]

"""
lowercased tokens - for use w/ word embeddings
in: string
out: string list
"""
def get_tokens(s):
    return [lower(t) for t in nltk.word_tokenize(s)]

"""
Cleans a string: trimming, removing non-alphanumeric
in: string
out: string
"""
def clean(s, lower = True):
    if lower:
        return re.sub('[\W_]+', ' ', s, flags = re.UNICODE).lower()
    else:
        return re.sub('[\W_]+', ' ', s, flags = re.UNICODE)

"""
Removes stopwords from a list of tokens
in: string list
out: string list
"""
def remove_stopwords(l):
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

"""
Function for getting lemmatized tokens in one step
in: string
out: string list
"""
def get_clean_tokens(body):
    clean_body = clean(body)
    tokens = get_tokenized_lemmas(clean_body)
    clean_tokens = remove_stopwords(tokens)
    return clean_tokens

"""
in: row number, dataframe (train_stances)
out: string of headline
"""
def get_headline(n, df):
    return (df.iloc[n])["Headline"]

"""
in: body ID, dataframe (train_bodies)
out: string of body text
"""
# get_body(6, train_bodies)
def get_body(n, df):
    return df.loc[lambda x: x["Body ID"] == n, "articleBody"].item()

"""
outs sentiment for single sentence
"""
def get_sentiment(sentence):
    return analyzer.polarity_scores(sentence)

"""
average of sentiment scores for each sentence
TODO: add weightings for sentences based on importance
{"pos", "neg", "neu", "compound"}

in: string w/ multiple sentences
out: dictionary object with sentiment scores for each category
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
helper function for IDF's
in: 2d list of strings
out: dict of string -> idf score (float)
"""
def build_idf_tokens(corpus, pos_tags = None):
    num_docs = len(corpus)
    vocabulary = Counter()
    for wordlist in list(corpus):
        if pos_tags is None:
            filtered_tokens = wordlist
        else:
            pos = nltk.pos_tag(wordlist)
            filtered_tokens = [x[0] for x in pos if (x[1] in pos_tags)]
        for i in set(filtered_tokens):
            vocabulary[i]+=1
    idf = {}
    for i in vocabulary:
        idf[i] = np.log(num_docs/(vocabulary[i]))
    return idf

"""
builds IDF from the headlines and bodies included in a stances df
NOTE: THIS IS VERY SLOW
example: idf = preprocessing.build_idf(stances_tr, ["NN","NNP", "NNPS", "NNS"])

in: body df, stances df (containing list of bodies to extract)
out: dict of string -> idf score (float)
"""
def build_idf(bodies, stances, pos_tags = None):
    corpus = [get_body(x, bodies) for x in set(stances['Body ID'])] + list(stances['Headline'])
    tokenized_corpus = [get_clean_tokens(x) for x in corpus]
    return build_idf_tokens(tokenized_corpus, pos_tags)

"""
get average tf-idf score of tokens in sentence
if no idf, then use tf score
must provide avg_idf if using idf!

in: list of tokens, tf dictionary, idf dictionary, average of idf
out: float
"""  
def score_sentence(sentence, tf, idf = None, avg_idf = None):
    acc = 0
    if idf == None:
        for token in sentence:
            acc+=tf[token]
    else:
        for token in sentence:
            if token in idf:
                acc+=tf[token]*idf[token]
            else:
                acc+=tf[token]*avg_idf
    return acc/len(sentence)

"""
extract metadata from sentence/body of text
in: string
out: dict
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

"""
extracts metadata from article body - READ COMMENTS FOR MORE INFO!
in: string, idf dictionary (string->float) [optional]
out: mixed dictionary

example usage:
process_body(get_body(6, train_bodies))
"""

def process_body(body, idf = None):
    sentences = list(nltk.tokenize.sent_tokenize(body))
    clean_sentences = [get_clean_tokens(s) for s in sentences] #sentences are tokenized
    clean_sentences = [s for s in clean_sentences if len(s) > 3]
    clean_tokens = [token for sentence in clean_sentences for token in sentence]
    body_length = len(clean_tokens)
    
    #look at first sentence of article
    first_sentence = sentences[0]
    first_sentence_data = process_sentence(first_sentence)
    
    #extracting bigrams and trigrams
    bigram = list(nltk.bigrams(clean_tokens))
    bigram_str = [x[0]+' '+x[1] for x in bigram]
    trigram = list(nltk.trigrams(clean_tokens))
    trigram_str = [x[0]+' '+x[1]+' '+x[2] for x in trigram]
    
    pos = nltk.pos_tag(clean_tokens)

    #count of each tag type (dict), counter for word (tf)
    tags_count = Counter([x[1] for x in pos])
    word_count = Counter([x[0] for x in pos])
    
    #list of words that belong to that part of speech
    nouns = [x[0] for x in pos if is_noun(x[1])]
    verbs = [x[0] for x in pos if is_verb(x[1])]
    adjectives = [x[0] for x in pos if is_adjective(x[1])]
    adverbs = [x[0] for x in pos if is_adverb(x[1])]

    num_nouns = len(nouns)
    num_verbs = len(verbs)
    doc_len = len(clean_tokens)

    n_counter = Counter(nouns)
    v_counter = Counter(verbs)
    b_counter = Counter(bigram)
    t_counter = Counter(trigram)

    #common words are highest scoring IDF (or TF if IDF not available)
    #significant sentence - sentence with highest average token IDF score
    #if no IDF use TF (which is not that good)
    if idf == None:
        common_nouns = [x[0] for x in n_counter.most_common(5)]
        common_verbs = [x[0] for x in v_counter.most_common(5)]
        #this is really shitty
        sentence_importance = [(s,score_sentence(s, word_count)) for s in clean_sentences]
        most_significant_sentence, sentence_score = list(sorted(sentence_importance, key = lambda x: x[1]))[-1]
        most_significant_sentence_data = process_sentence(' '.join(most_significant_sentence))

    else: 
        avg_idf = float(sum(idf.values())) / len(idf)
        n_tfidf, v_tfidf = {},{}
        for n in n_counter:
            n_tfidf[n] = (n_counter[n]/doc_len)*(idf[n] if n in idf else avg_idf)
        for v in v_counter:
            v_tfidf[v] = (v_counter[v]/doc_len)*(idf[v] if v in idf else avg_idf) 
        common_nouns = sorted(n_tfidf, key=n_tfidf.get, reverse=True)[:10]
        common_verbs = sorted(v_tfidf, key=v_tfidf.get, reverse=True)[:10]
        
        sentence_importance = [(s,score_sentence(s, word_count, idf, avg_idf)) for s in clean_sentences]
        most_significant_sentence, sentence_score = list(sorted(sentence_importance, key = lambda x: x[1]))[-1]
        most_significant_sentence_data = process_sentence(' '.join(most_significant_sentence))

    #no idf for bigrams/trigrams :(
    common_bigrams = [x[0] for x in b_counter.most_common(5)]
    common_trigrams = [x[0] for x in t_counter.most_common(5)]

    n_adj = len(adjectives)
    n_adv = len(adverbs)
    #breakdown of porportion of regular/comparative/superlative adverbs as a tuple (in that order)
    if n_adj!=0:
        adj_types = (tags_count['JJ']/n_adj,tags_count['JJR']/n_adj,tags_count['JJS']/n_adj)
    else:
        adj_types = (0,0,0)
    if n_adv!=0:
        adv_types = (tags_count['RB']/n_adv,tags_count['RBR']/n_adv,tags_count['RBS']/n_adv)
    else:
        adv_types = (0,0,0)

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
        "significant_sentence": most_significant_sentence_data,
        "adj_types": adj_types,
        "adv_types": adv_types,
        "vocabulary": set(clean_tokens),
        "common_nouns": common_nouns,
        "common_verbs": common_verbs,
        "common_bigrams": common_bigrams,
        "common_trigrams": common_trigrams
    }

"""
in: df of bodies, idf dict (string->float) [optional]
out: dict with k=bodyid and v=dict of bodyinfo as per process_body
"""
def process_bodies(df, idf = None):
    body_info = {}
    ids = list(df["Body ID"])
    for i in range(len(ids)):
        if i%100 == 0 and i!=0:
            print("processed "+str(i))
        body_info[ids[i]]= process_body(get_body(ids[i], df), idf)
    print("done! processed "+ str(len(ids)))
    return body_info

"""
in: df of bodies, df of stances, fraction of bodies you want to be in training set
out: 2 df's of stances, first is training second is test; 2 lists of

this ensures no overlap of bodies between train and test sets, exactly like actual testing

example usage: stances_tr, stances_val = preprocessing.train_test_split(train_bodies, train_stances)
"""
def train_test_split(bodies, stances, split = 0.8):
    idx = np.random.permutation(np.arange(len(bodies))) 
    bodies = bodies.values[idx]
    train = int(len(bodies)*0.8)
    bodies_tr = set([i[0] for i in bodies[:train]])
    bodies_val = set([i[0] for i in bodies[train:]])
    stances_tr = stances.loc[stances["Body ID"].isin(bodies_tr), :]
    stances_val = stances.loc[stances["Body ID"].isin(bodies_val), :]
    return stances_tr, stances_val

"""
return a dictionary of features
"""
def get_feats(data, body_dict):
    headline, body_id = data[0],int(data[1])
    headline_data = process_sentence(headline)

    shared_common_nouns = len(set(headline_data['nouns']).intersection(set(body_dict[body_id]['common_nouns'])))
    shared_common_verbs = len(set(headline_data['verbs']).intersection(set(body_dict[body_id]['common_verbs'])))
    shared_tokens = len(set(headline_data['tokens']).intersection(set(body_dict[body_id]['tokens'])))
    shared_bigrams = len(set(headline_data['bigrams']).intersection(set(body_dict[body_id]['common_bigrams'])))
    shared_trigrams = len(set(headline_data['trigrams']).intersection(set(body_dict[body_id]['common_trigrams'])))

    shared_nouns_first = len(set(headline_data['nouns']).intersection(set(body_dict[body_id]['first_sentence']['nouns'])))
    shared_verbs_first = len(set(headline_data['verbs']).intersection(set(body_dict[body_id]['first_sentence']['verbs'])))
    shared_bigrams_first = len(set(headline_data['bigrams']).intersection(set(body_dict[body_id]['first_sentence']['bigrams'])))
    shared_trigrams_first = len(set(headline_data['trigrams']).intersection(set(body_dict[body_id]['first_sentence']['trigrams'])))
    shared_tokens_first = len(set(headline_data['tokens']).intersection(set(body_dict[body_id]['first_sentence']['tokens'])))

    shared_nouns_sig = len(set(headline_data['nouns']).intersection(set(body_dict[body_id]['significant_sentence']['nouns'])))
    shared_verbs_sig = len(set(headline_data['verbs']).intersection(set(body_dict[body_id]['significant_sentence']['verbs'])))
    shared_bigrams_sig = len(set(headline_data['bigrams']).intersection(set(body_dict[body_id]['significant_sentence']['bigrams'])))
    shared_trigrams_sig = len(set(headline_data['trigrams']).intersection(set(body_dict[body_id]['significant_sentence']['trigrams'])))
    shared_tokens_sig = len(set(headline_data['tokens']).intersection(set(body_dict[body_id]['significant_sentence']['tokens'])))
    
    sentiment_diff = {
        "pos": headline_data['sentiment']['pos']-body_dict[body_id]['sentiment']['pos'],
        "neg": headline_data['sentiment']['neg']-body_dict[body_id]['sentiment']['neg'],
        "neu": headline_data['sentiment']['neu']-body_dict[body_id]['sentiment']['neu'],
        "compound": headline_data['sentiment']['compound']-body_dict[body_id]['sentiment']['compound']
    }
    sentiment_diff_first = {
        "pos": headline_data['sentiment']['pos']-body_dict[body_id]['first_sentence']['sentiment']['pos'],
        "neg": headline_data['sentiment']['neg']-body_dict[body_id]['first_sentence']['sentiment']['neg'],
        "neu": headline_data['sentiment']['neu']-body_dict[body_id]['first_sentence']['sentiment']['neu'],
        "compound": headline_data['sentiment']['compound']-body_dict[body_id]['first_sentence']['sentiment']['compound']
    }
    sentiment_diff_sig = {
        "pos": headline_data['sentiment']['pos']-body_dict[body_id]['significant_sentence']['sentiment']['pos'],
        "neg": headline_data['sentiment']['neg']-body_dict[body_id]['significant_sentence']['sentiment']['neg'],
        "neu": headline_data['sentiment']['neu']-body_dict[body_id]['significant_sentence']['sentiment']['neu'],
        "compound": headline_data['sentiment']['compound']-body_dict[body_id]['significant_sentence']['sentiment']['compound']
    }

    return {
        'shared_nouns': shared_common_nouns,
        'shared_verbs': shared_common_verbs,
        'shared_bigrams': shared_bigrams,
        'shared_trigrams': shared_trigrams,
        'shared_tokens': shared_tokens,

        'shared_nouns_fst':shared_nouns_first,
        'shared_verbs_fst':shared_verbs_first,
        'shared_bigrams_fst':shared_bigrams_first,
        'shared_trigrams_fst':shared_trigrams_first,
        'shared_tokens_fst': shared_tokens_first,

        'shared_nouns_sig':shared_nouns_sig,
        'shared_verbs_sig':shared_verbs_sig,
        'shared_bigrams_sig':shared_bigrams_sig,
        'shared_trigrams_sig':shared_trigrams_sig,
        'shared_tokens_sig': shared_tokens_sig,

        'sentiment_pos': sentiment_diff['pos'],
        'sentiment_neg': sentiment_diff['neg'],
        'sentiment_neu': sentiment_diff['neu'],
        'sentiment_compound':sentiment_diff_first['compound'],

        'sentiment_pos_fst': sentiment_diff_first['pos'],
        'sentiment_neg_fst': sentiment_diff_first['neg'],
        'sentiment_neu_fst': sentiment_diff_first['neu'],
        'sentiment_compound_fst':sentiment_diff_first['compound'],

        'sentiment_pos_sig': sentiment_diff_sig['pos'],
        'sentiment_neg_sig': sentiment_diff_sig['neg'],
        'sentiment_neu_sig': sentiment_diff_sig['neu'],
        'sentiment_compound_sig':sentiment_diff_sig['compound']
    }