import helpers
import word_embeddings
import feature_engineering
import pickle
import queue
import spacy
from scipy import spatial
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn import feature_extraction
import numpy as np
from nltk import tokenize
import nltk
import re
"""
FUNCTIONS FOR PREPROCESSING AND FEATURE ENGINEERING - pending reorganization
"""


"""
Helpers borrowed from baseline implementation in feature_engineering.py
"""

import re
import nltk
from nltk import tokenize
import numpy as np
from sklearn import feature_extraction

from collections import Counter
from feature_engineering import FeatureEngineering
from helpers import Helpers
from word_embeddings import WordEmbeddings

class Preprocessing(Helpers, FeatureEngineering, WordEmbeddings):

    def __init__(self):
        Helpers.__init__(self)
        WordEmbeddings.__init__(self)
        FeatureEngineering.__init__(self)

    """
    in: row number, dataframe (train_stances)
    out: string of headline
    """

    def get_headline(self, n, df):
        return (df.iloc[n])["Headline"]


    """
    in: body ID, dataframe (train_bodies)
    out: string of body text
    example: get_body(6, train_bodies)
    """


    def get_body(self, n, df):
        return df.loc[lambda x: x["Body ID"] == n, "articleBody"].item()


    """
    builds IDF from the headlines and bodies included in a stances df
    NOTE: THIS IS VERY SLOW
    example: idf = preprocessing.build_idf(stances_tr, ["NN","NNP", "NNPS", "NNS"])

    in: body df, stances df (containing list of bodies to extract)
    out: dict of string -> idf score (float) with one special entry (_avg -> average idf score)
    """


    def build_idf(self, bodies, stances, pos_tags=None):
        corpus = [self.get_body(x, bodies) for x in set(
            stances['Body ID'])] + list(stances['Headline'])
        tokenized_corpus = [self.get_clean_tokens(x) for x in corpus]
        idf = self.build_idf_tokens(tokenized_corpus, pos_tags)
        idf["_avg"] = float(sum(idf.values())) / len(idf)
        return idf


    """
    extracts metadata from article body - READ COMMENTS FOR MORE INFO!
    in: string, idf dictionary (string->float) [optional]
    out: mixed dictionary

    example usage:
    process_body(get_body(6, train_bodies))
    """


    def process_body(self, body, idf=None):
        sentences = list(nltk.tokenize.sent_tokenize(body))
        # sentences are tokenized
        clean_sentences = [self.get_clean_tokens(s) for s in sentences]
        clean_sentences = [s for s in clean_sentences if len(s) > 3]
        clean_tokens = [
            token for sentence in clean_sentences for token in sentence]
        body_length = len(clean_tokens)

        # look at first sentence of article
        first_sentence = sentences[0]
        first_sentence_data = self.process_sentence(first_sentence)

        # extracting bigrams
        bigram = list(nltk.bigrams(clean_tokens))
        bigram_str = [x[0]+' '+x[1] for x in bigram]

        pos = nltk.pos_tag(clean_tokens)

        # count of each tag type (dict), counter for word (tf)
        tags_count = Counter([x[1] for x in pos])
        word_count = Counter([x[0] for x in pos])

        # list of words that belong to that part of speech
        nouns = [x[0] for x in pos if self.is_noun(x[1])]
        verbs = [x[0] for x in pos if self.is_verb(x[1])]
        adjectives = [x[0] for x in pos if self.is_adjective(x[1])]
        adverbs = [x[0] for x in pos if self.is_adverb(x[1])]

        doc_len = len(clean_tokens)

        n_counter = Counter(nouns)
        v_counter = Counter(verbs)
        b_counter = Counter(bigram)
        token_counter = Counter(clean_tokens)

        # common words are highest scoring IDF (or TF if IDF not available)
        # significant sentence - sentence with highest average token IDF score
        # if no IDF use TF (which is not that good)
        if idf == None:
            common_nouns = [x[0] for x in n_counter.most_common(5)]
            common_verbs = [x[0] for x in v_counter.most_common(5)]
            common_tokens = [x[0] for x in token_counter.most_common(5)]
            # this is really shitty
            sentence_importance = [(s, self.score_sentence(s, word_count))
                                for s in clean_sentences]
            most_significant_sentence, sentence_score = list(
                sorted(sentence_importance, key=lambda x: x[1]))[-1]
            most_significant_sentence_data = self.process_sentence(
                ' '.join(most_significant_sentence))

        else:
            avg_idf = idf["_avg"]
            n_tfidf, v_tfidf, t_tfidf = {}, {}, {}
            for n in n_counter:
                n_tfidf[n] = (n_counter[n]/doc_len) * \
                    (idf[n] if n in idf else avg_idf)
            for v in v_counter:
                v_tfidf[v] = (v_counter[v]/doc_len) * \
                    (idf[v] if v in idf else avg_idf)
            for t in token_counter:
                t_tfidf[t] = (token_counter[t]/doc_len) * \
                    (idf[t] if t in idf else avg_idf)
            common_nouns = sorted(n_tfidf, key=n_tfidf.get, reverse=True)[:5]
            common_verbs = sorted(v_tfidf, key=v_tfidf.get, reverse=True)[:5]
            common_tokens = sorted(t_tfidf, key=t_tfidf.get, reverse=True)[:5]

            sentence_importance = [
                (s, self.score_sentence(s, word_count, idf)) for s in clean_sentences]
            most_significant_sentence, sentence_score = list(
                sorted(sentence_importance, key=lambda x: x[1]))[-1]
            most_significant_sentence_data = self.process_sentence(
                ' '.join(most_significant_sentence))

        # no idf for bigrams increase "common" count to 10
        common_bigrams = [x[0] for x in b_counter.most_common(10)]

        n_adj = len(adjectives)
        n_adv = len(adverbs)
        # breakdown of porportion of regular/comparative/superlative adverbs as a tuple (in that order)
        if n_adj != 0:
            adj_types = (tags_count['JJ']/n_adj,
                        tags_count['JJR']/n_adj, tags_count['JJS']/n_adj)
        else:
            adj_types = (0, 0, 0)
        if n_adv != 0:
            adv_types = (tags_count['RB']/n_adv,
                        tags_count['RBR']/n_adv, tags_count['RBS']/n_adv)
        else:
            adv_types = (0, 0, 0)

        vader_sentiment = self.sentiment_multi(body)

        return {
            "tokens": clean_tokens,
            "bigrams": bigram_str,
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
            "common_tokens": common_tokens,
            "common_nouns": common_nouns,
            "common_verbs": common_verbs,
            "common_bigrams": common_bigrams,
        }


    """
    in: df of bodies, idf dict (string->float) [optional]
    out: dict with k=bodyid and v=dict of bodyinfo as per process_body

    NOTE: THIS IS VERY SLOW
    """


    def process_bodies(self, df, idf=None):
        body_info = {}
        ids = list(df["Body ID"])
        for i in range(len(ids)):
            if i % 100 == 0 and i != 0:
                print("processed "+str(i))
            body_info[ids[i]] = self.process_body(self.get_body(ids[i], df), idf)
        print("done! processed " + str(len(ids)))
        return body_info


    """
    in: df of bodies, df of stances, fraction of bodies you want to be in training set
    out: 2 df's of stances, first is training second is test; 2 lists of

    this ensures no overlap of bodies between train and test sets, exactly like actual testing

    example usage: stances_tr, stances_val = preprocessing.train_test_split(train_bodies, train_stances)
    """


    def train_test_split(self, bodies, stances, split=0.8):
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
    see usage example:

    data - [headline:str, body_id:str/int]
    body_dict - dictionary of processed bodies
    idf - dictionary of idf scores

    NOTE: THIS IS VERY SLOW
    """
    def get_feats(self, data, body_dict, idf = None):
        return self.get_feats(data, body_dict, idf)
